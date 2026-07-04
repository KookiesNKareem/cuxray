"""Minimal ELF64 reader for cubins — no dependencies, no execution.

Gives cuxray three cheap facts without invoking nvdisasm:
  - machine():   e_machine (EM_CUDA=190 → cubin; otherwise host ELF)
  - sm_arch():   architecture from e_flags bits 8..15 (0x5a → sm_90,
                 0x78 → sm_120; verified against fixture cubins). The 'a'
                 suffix is not recoverable here — full reports refine the
                 arch from nvdisasm's `.target` line.
  - functions(): (symbol_index, name) for STT_FUNC symbols, in symtab order.
                 Symbol indices feed `nvdisasm -fun i,j,...` to restrict
                 disassembly to matching kernels (38s → 1.4s on an 8 MB
                 production Marlin cubin).
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional

EM_CUDA = 190

_SHT_SYMTAB = 2
_STT_FUNC = 2


def machine(data: bytes) -> Optional[int]:
    if len(data) < 0x40 or data[:4] != b"\x7fELF":
        return None
    return struct.unpack_from("<H", data, 18)[0]


def sm_arch(data: bytes) -> Optional[str]:
    if machine(data) != EM_CUDA:
        return None
    e_flags = struct.unpack_from("<I", data, 0x30)[0]
    sm = (e_flags >> 8) & 0xFF
    return f"sm_{sm}" if sm else None


def _sections(data: bytes) -> list[dict]:
    shoff = struct.unpack_from("<Q", data, 0x28)[0]
    shentsize = struct.unpack_from("<H", data, 0x3A)[0]
    shnum = struct.unpack_from("<H", data, 0x3C)[0]
    secs = []
    for i in range(shnum):
        off = shoff + i * shentsize
        name, typ, _flags, _addr, offset, size, link, _info, _align, entsize = (
            struct.unpack_from("<IIQQQQIIQQ", data, off)
        )
        secs.append({"typ": typ, "offset": offset, "size": size,
                     "link": link, "entsize": entsize})
    return secs


def functions(data: bytes) -> list[tuple[int, str]]:
    """All STT_FUNC symbols as (symbol_index, name)."""
    if machine(data) is None:
        return []
    out: list[tuple[int, str]] = []
    secs = _sections(data)
    for s in secs:
        if s["typ"] != _SHT_SYMTAB or not s["entsize"]:
            continue
        strtab = secs[s["link"]]
        for i in range(s["size"] // s["entsize"]):
            off = s["offset"] + i * s["entsize"]
            nameoff, info, _other, _shndx, _value, _size = struct.unpack_from(
                "<IBBHQQ", data, off
            )
            if info & 0xF != _STT_FUNC:
                continue
            ns = strtab["offset"] + nameoff
            end = data.index(b"\0", ns)
            out.append((i, data[ns:end].decode(errors="replace")))
        break  # first (real) .symtab only; .nv.merc.symtab shadows it
    return out
