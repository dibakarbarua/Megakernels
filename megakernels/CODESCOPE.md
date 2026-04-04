# megakernels/ Python Package Code Scope

## Purpose

This package is the Python control plane for the repo.

It is responsible for:

- defining the reference Llama model
- loading and reshaping weights
- building instruction schedules
- choosing execution backends
- providing generation loops for PyTorch, Python VM, and CUDA megakernel modes

## High-level file map

- `llama.py`
  Reference model, Hugging Face weight loading, RoPE handling, stacked-parameter
  creation, and KV cache setup.
- `model_types.py`
  Shared dataclasses such as `BatchState`, `ModelOutput`, and
  `ExtraModelConfig`.
- `instructions.py`
  Base instruction serialization and the common global-buffer dataclass.
- `scheduler.py`
  Generic DAG structures, SM assignment helpers, and instruction tensorization.
- `dispatch.py`
  Mode switchboard that chooses latency vs throughput schedule builders and
  interpreters.
- `generators.py`
  Decode loops for PyTorch, Python VM, and CUDA megakernel execution.
- `python_vm.py`
  Minimal reference interpreter infrastructure.
- `mk.py`
  Dynamic Python loader for compiled megakernel extension modules.
- `utils.py`
  Helper functions for weight loading, tensor-parallel slicing, and device SM
  count discovery.
- `demos/`
  Mode-specific instruction sets and interpreters.
- `scripts/`
  User-facing entrypoints.

## End-to-end flow inside this package

### 1. Load model and stack parameters

`llama.py` loads Hugging Face weights, optionally interleaves RoPE layout for
the latency kernel, stacks layer weights into contiguous tensors, and allocates
the global KV caches.

This is where the object-oriented PyTorch module graph gets flattened into
megakernel-friendly arrays.

### 2. Build mode-specific globals and instructions

`dispatch.py` picks either the latency or throughput builder.

That builder:

- allocates global activation/buffer tensors
- creates instruction objects for the full model forward
- returns a `Schedule`

### 3. Assign instructions to SMs

`scheduler.py` assigns the instruction DAG to SM queues and serializes each
instruction to a 32-int record.

The resulting buffers are stored directly on `schedule.globs`.

### 4. Run one decode step

`generators.py` embeds the current token, writes `hidden_states`, sets `pos_id`,
and then either:

- runs the PyTorch model
- runs the Python VM over instruction objects
- calls the compiled CUDA megakernel

### 5. Extract logits and pick the next token

PyTorch mode gets its output directly from the model. The VM and CUDA modes
write into the shared logits buffer, and Python does the token selection.

## Core dependency graph

- `scripts/*` -> `dispatch.py`, `generators.py`, `llama.py`
- `dispatch.py` -> `demos/{latency,throughput}/*`
- `generators.py` -> `llama.py`, `scheduler.py`, interpreters
- `llama.py` -> `model_types.py`, `utils.py`
- `demos/{latency,throughput}` -> `instructions.py`, `python_vm.py`,
  `scheduler.py`, `llama.py`

## Reading advice

Read these next:

1. `scripts/CODESCOPE.md`
2. `demos/CODESCOPE.md`

Then choose:

- `demos/latency/CODESCOPE.md` for the main shipped path
- `demos/throughput/CODESCOPE.md` for the batched variant
