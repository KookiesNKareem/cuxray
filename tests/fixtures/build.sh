#!/usr/bin/env bash
# Build cuxray test fixtures: cubins + recorded tool outputs.
# Requires nvcc/ptxas/nvdisasm/cuobjdump on PATH (no GPU needed).
# Usage: tests/fixtures/build.sh [outdir]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/src"
BIN="${1:-$HERE/bin}"
REC="$HERE/recorded"
mkdir -p "$BIN" "$REC"

ARCHES=(sm_90 sm_120a)

compile() { # name, arch, extra nvcc flags...
    local name="$1" arch="$2"; shift 2
    local out="$BIN/${name}.${arch}.cubin"
    nvcc -cubin "-arch=$arch" -lineinfo --resource-usage -o "$out" "$@" \
        "$SRC/${name}.cu" 2> "$REC/ptxas_v.${name}.${arch}.txt"
    cuobjdump --dump-resource-usage "$out" > "$REC/resusage.${name}.${arch}.txt"
}

for arch in "${ARCHES[@]}"; do
    compile saxpy         "$arch"
    compile tiled_matmul  "$arch"
    compile launch_bounds "$arch"
    compile spill         "$arch" -maxrregcount 32
done

# Multi-arch host object (fatbin embedded in a host ELF) for extraction tests
nvcc -c -lineinfo \
    -gencode arch=compute_90,code=sm_90 \
    -gencode arch=compute_120a,code=sm_120a \
    -o "$BIN/multiarch.o" "$SRC/launch_bounds.cu"
cuobjdump --list-elf "$BIN/multiarch.o" > "$REC/listelf.multiarch.txt"

# Recorded nvdisasm outputs (parser fixtures) for representative kernels
for arch in "${ARCHES[@]}"; do
    for name in saxpy spill tiled_matmul; do
        cub="$BIN/${name}.${arch}.cubin"
        nvdisasm -c -gi "$cub"        > "$REC/nvdisasm_gi.${name}.${arch}.txt"
        nvdisasm -c -plr -gi "$cub"   > "$REC/nvdisasm_plr.${name}.${arch}.txt"
        nvdisasm -cfg "$cub"          > "$REC/nvdisasm_cfg.${name}.${arch}.dot"
    done
done

# ELF header dumps for arch-detection fixtures (full output; head would SIGPIPE)
cuobjdump -elf "$BIN/saxpy.sm_90.cubin" > "$REC/elfhdr.saxpy.sm_90.txt"
cuobjdump -elf "$BIN/saxpy.sm_120a.cubin" > "$REC/elfhdr.saxpy.sm_120a.txt"

echo "fixtures built into $BIN, recordings in $REC"
