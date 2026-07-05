"""Minimal ELF64 reader for cubins — no dependencies, no execution.

Gives cuxray three cheap facts without invoking nvdisasm:
  - machine():   e_machine (EM_CUDA=190 → cubin; otherwise host ELF)
  - sm_arch():   architecture from e_flags (bits 8-15 on CUDA 13.x tools,
                 low byte on 12.x; disambiguated by SM plausibility). The
                 'a' suffix is not recoverable here — full reports refine
                 the arch from nvdisasm's `.target` line.
  - functions(): (symbol_index, name) for STT_FUNC symbols, in symtab order.
                 Symbol indices feed `nvdisasm -fun i,j,...` to restrict
                 disassembly to matching kernels.
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


_VALID_SM = {50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 87, 89, 90,
             100, 101, 103, 110, 120, 121}


def sm_arch(data: bytes) -> Optional[str]:
    """SM number from e_flags. CUDA 13.x encodes it in bits 8-15; CUDA 12.x
    and earlier in the low byte — disambiguated by plausibility."""
    if machine(data) != EM_CUDA:
        return None
    e_flags = struct.unpack_from("<I", data, 0x30)[0]
    for cand in ((e_flags >> 8) & 0xFF, e_flags & 0xFF):
        if cand in _VALID_SM:
            return f"sm_{cand}"
    return None


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


_EIFMT_SVAL = 0x04
_EIATTR_MAX_THREADS = 0x05  # __launch_bounds__ / .maxntid (upper bound)
_EIATTR_REQNTID = 0x10      # .reqntid (exact block shape)
# Both derived empirically against cuobjdump --dump-elf decode on fixtures
# (launch_bounds.sm_90.cubin → MAX_THREADS [256,1,1]; hand-written .reqntid
# PTX → REQNTID [96,2,1]).


def launch_dims(data: bytes) -> dict[str, dict]:
    """Per-kernel block-shape metadata from .nv.info.<kernel> sections.

    Returns {kernel: {"reqntid": (x,y,z)|None, "maxntid": (x,y,z)|None}}.
    reqntid is the exact launch shape; maxntid an upper bound (__launch_bounds__).
    """
    import struct as _struct
    if machine(data) != EM_CUDA:
        return {}
    shoff = _struct.unpack_from("<Q", data, 0x28)[0]
    shentsize = _struct.unpack_from("<H", data, 0x3A)[0]
    shnum = _struct.unpack_from("<H", data, 0x3C)[0]
    shstrndx = _struct.unpack_from("<H", data, 0x3E)[0]
    secs = _sections(data)
    shstr = secs[shstrndx]

    def sec_name(nameoff: int) -> str:
        s = shstr["offset"] + nameoff
        return data[s:data.index(b"\0", s)].decode(errors="replace")

    out: dict[str, dict] = {}
    for i in range(shnum):
        off = shoff + i * shentsize
        nameoff = _struct.unpack_from("<I", data, off)[0]
        name = sec_name(nameoff)
        if not name.startswith(".nv.info."):
            continue
        kernel = name[len(".nv.info."):]
        raw = data[secs[i]["offset"]:secs[i]["offset"] + secs[i]["size"]]
        entry = out.setdefault(kernel, {"reqntid": None, "maxntid": None})
        p = 0
        while p + 4 <= len(raw):
            fmt, attr = raw[p], raw[p + 1]
            if fmt == _EIFMT_SVAL:
                sz = _struct.unpack_from("<H", raw, p + 2)[0]
                if attr in (_EIATTR_MAX_THREADS, _EIATTR_REQNTID) and sz >= 12:
                    dims = tuple(
                        _struct.unpack_from("<I", raw, p + 4 + k)[0]
                        for k in (0, 4, 8)
                    )
                    key = "reqntid" if attr == _EIATTR_REQNTID else "maxntid"
                    entry[key] = dims
                p += 4 + sz
            else:
                p += 4  # NVAL/BVAL/HVAL are all 4-byte records
    return out


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
