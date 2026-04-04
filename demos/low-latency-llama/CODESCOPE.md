# low-latency-llama Code Scope

## Purpose

This directory contains the model-specific CUDA side of the low-latency Llama
decode demo.

Its job is to answer one question:

"Given a serialized instruction stream plus global model buffers, how does each
opcode execute on the GPU?"

## Files and roles

- `llama.cuh`
  Defines the Llama-specific global-memory layout, compile-time model
  dimensions, opcodes, hardware-specific SM count, and the `llama_1b_globals`
  type expected by the bound kernel.
- `llama.cu`
  Binds the compiled kernel to Python as `mk_llama` and registers the concrete
  operator set used by the interpreter.
- `matvec_pipeline.cuh`
  Shared pipeline template used by most matvec-style fused operators. This is
  the main reusable helper for load/compute/store overlap.
- `utils.cuh`
  Small math helpers such as RMS norm and cross-warp reduction of partial
  matvec outputs.
- `rms_matvec_rope_append.cu`
  Opcode 1. Fused attention input path:
  RMS norm -> QKV projection -> RoPE on Q and K -> append K/V to cache.
- `attention_partial.cu`
  Opcode 2. Partial attention decode for one KV head group.
- `attention_reduction.cu`
  Opcode 3. Reduction tree for combining multiple attention partials.
  Present for generality, often bypassed in the shipped latency schedule.
- `matvec_adds.cu`
  Generic template used to implement:
  - O projection + residual
  - down projection + residual
- `upgate.cu`
  Opcode 5. RMS norm + up projection + gate projection + SiLU gating.
- `rms_lm_head.cu`
  Opcode 7. Final RMS norm + LM head projection.
- `Makefile`
  Builds the `mk_llama` extension.

## How it fits into the end-to-end flow

### Inputs from Python

Python passes the following categories of buffers into `mk_llama`:

- serialized instructions
- barrier counters
- stacked weights
- KV caches
- RoPE tables
- activation buffers
- scalar state like `pos_id`, `attn_scale`, and `rms_norm_eps`

Those buffers are prepared by:

- `megakernels/demos/latency/scheduler.py`
- `megakernels/scheduler.py`
- `megakernels/demos/latency/mk.py`

### Outputs back to Python

The main side effects of this kernel are:

- updated KV cache entries
- updated `hidden_states`
- populated `logits`
- updated barrier counters and optional timings

Python then consumes `logits` and chooses the next token.

## Important architectural ideas

### 1. Model shape is hardcoded here

`llama.cuh` hardcodes the current Llama variant dimensions. Porting to another
model means this file is one of the first places that must change.

### 2. Ops are interpreter plugins

Every `.cu` file defines a struct with nested `controller`, `loader`,
`launcher`, `consumer`, and `storer` behaviors. The generic runtime dispatches
to those nested hooks based on opcode.

### 3. Most math ops use a common pipeline

`matvec_pipeline.cuh` is the main shared implementation for:

- loading activation and weight tiles
- assigning shared-memory pages
- overlapping load, compute, and store stages
- accumulating partial results across consumer warps

This is where much of the "no bubbles" execution strategy becomes concrete.

### 4. Attention is more specialized than matvec ops

`attention_partial.cu` and `attention_reduction.cu` are more bespoke because
attention needs:

- KV streaming across sequence length
- grouped-query attention handling
- LSE accumulation for partial reductions
- different shared-memory layouts than plain matvec operators

## Dependency map

This directory depends on:

- `include/CODESCOPE.md` runtime primitives
- ThunderKittens tensor/TMA abstractions

This directory is depended on by:

- `megakernels/demos/latency/mk.py`
- `megakernels/mk.py`
- `megakernels/scripts/generate.py`
- `megakernels/scripts/llama_repl.py`

## Porting checklist

If you are adapting this directory for another model or ASIC, the main moving
parts are:

- opcodes and globals layout in `llama.cuh`
- Python scheduler lowering that emits matching instructions
- per-op data layouts and dependency counts
- any model-specific assumptions such as RoPE layout or GQA ratio
