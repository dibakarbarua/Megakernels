# demos/ Code Scope

## Purpose

`demos/` contains model-specific CUDA implementations that plug into the generic
VM runtime in `include/`.

The key distinction is:

- `include/` defines the interpreter framework
- `demos/` defines the actual model operators and model-specific globals

## Directories

- `low-latency-llama/`
  The primary shipped CUDA demo. It binds a Llama-specific megakernel module
  named `mk_llama` and implements the fused decode operators.

## Dependency direction

`demos/*` depends on:

QUESTION: Need to understand the exact nature of operation for any one instruction

- `include/megakernel.cuh`
- `include/config.cuh`
- `include/util.cuh`
- the worker-loop wrappers in `include/{consumer,loader,storer,launcher}.cuh`

`demos/*` is depended on by:

- Python interpreters such as `megakernels/demos/latency/mk.py`
- the user-facing scripts in `megakernels/scripts/`

## Reading advice

For the main path, read `low-latency-llama/CODESCOPE.md` next.
