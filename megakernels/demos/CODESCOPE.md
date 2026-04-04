# megakernels/demos Code Scope

## Purpose

This directory contains the Python-side lowering logic for different execution
settings.

Both subdirectories implement the same broad interfaces:

- an instruction set
- a globals dataclass
- a schedule builder
- a Python VM interpreter mapping
- a CUDA interpreter wrapper

The difference is what they optimize for.

## Directories

- `latency/`
  Single-token decode path. This is the most important directory for
  understanding the shipped low-latency Llama megakernel.
- `throughput/`
  Batched path with a different instruction decomposition and barrier layout.

## Shared pattern

Each mode follows this pipeline:

1. Define mode-specific `Globals`
2. Define concrete instruction classes and opcodes
3. Build a schedule from the model
4. Provide a Python VM interpreter for semantic validation
5. Provide a CUDA wrapper that marshals tensors into the bound extension

## Dependency direction

These subdirectories depend on:

- `megakernels/llama.py`
- `megakernels/instructions.py`
- `megakernels/python_vm.py`
- `megakernels/scheduler.py`

They are depended on by:

- `megakernels/dispatch.py`
- `megakernels/scripts/*`

## Reading advice

Read `latency/CODESCOPE.md` first. It is the clearest expression of the core
decode architecture.
