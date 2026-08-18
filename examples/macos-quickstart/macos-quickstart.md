# Analyze a CUDA kernel on a Mac in two minutes

This example demonstrates how cuxray turns a real CUDA binary into an
actionable optimization. It does not require an NVIDIA GPU, the CUDA toolkit,
a Linux machine, or a repository checkout.

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

Docker Desktop and OrbStack also work. If either is already running, you do not
need Colima.

## 2. Analyze the example

Run:

```console
cuxray demo
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

For machine-readable output, run:

```console
cuxray demo --json
```

The bundled binary and its matching [`spill.cu`](spill.cu) source are also
available in the repository for deeper inspection.

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
