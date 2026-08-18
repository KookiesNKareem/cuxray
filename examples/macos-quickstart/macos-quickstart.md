# Analyze a CUDA kernel on a Mac in two minutes

This example demonstrates how to use cuxray from a real CUDA binary and to an actionable kernel
optimization. It does not require an NVIDIA GPU, the CUDA toolkit, or a Linux
machine.

The included kernel deliberately keeps too much per-thread state live. It was
compiled with a 32-register cap, forcing values into local memory. cuxray finds
those spills and identifies the hottest source line using only the compiled
binary.

## 1. Install cuxray

Install the lightweight macOS prerequisites once:

```console
brew install pipx colima docker
pipx ensurepath
pipx install cuxray
```

Open a new terminal if `pipx ensurepath` asks you to. If you do not already
have this repository checked out, get the example with:

```console
git clone https://github.com/KookiesNKareem/cuxray.git
cd cuxray
```

Docker Desktop and OrbStack also work. If either is already running, you do not
need Colima.

## 2. Analyze the example

From the root of this repository, run:

```console
cuxray advise tests/fixtures/bin/spill.sm_90.cubin --threads 256
```

On the first analysis, cuxray explains why it needs a Linux helper and asks
permission to start it:

```text
cuxray needs a lightweight Linux helper because NVIDIA's analysis tools are unavailable for macOS.
Start it with Colima now? [Y/n]
```

Press Enter. cuxray starts an isolated Colima profile and downloads its helper
image. Both are reused automatically on later commands.

The analysis then reports:

```text
spilly(float const*, float*, int, int)  (spill.sm_90.cubin)
  1. eliminate register spills  · high confidence · impact 1244
     154 spill stores / 155 loads to local memory (hottest at spill.cu:14);
     raise -maxrregcount or cut live state
     evidence: spill byte accounting (validated vs ptxas)
```

That is the optimization target: the accumulator array in
[`spill.cu`](spill.cu) is kept live across the inner loop, but the artificial
register cap forces part of it into local memory. cuxray identifies both the
problem and the hottest line without running the kernel.

For the underlying measurements, run:

```console
cuxray report tests/fixtures/bin/spill.sm_90.cubin --threads 256
```

The report includes the 32-register allocation, 620 bytes of spill stores, 624
bytes of spill loads, source-line attribution, and modeled occupancy.

## What happened behind the scenes?

The `cuxray` command still ran on your Mac. For commands requiring NVIDIA's
Linux-only binary-analysis utilities, it transparently mounted the current
directory into a small Linux container and returned the container's output and
exit code. It did not emulate or execute the CUDA kernel, and no GPU was used.

You can inspect the setup at any time:

```console
cuxray doctor
```

The helper can remain running for fast subsequent analyses. To stop it:

```console
colima --profile cuxray stop
```
