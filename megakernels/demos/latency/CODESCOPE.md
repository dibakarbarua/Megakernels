# latency Mode Code Scope

## Purpose

This directory contains the Python-side implementation of the low-latency decode
mode.

This is the main control-plane counterpart to
`demos/low-latency-llama/`.

## Files and roles

- `instructions.py`
  Defines the latency-mode instruction set, global activation buffers, and
  opcode numbering.
- `scheduler.py`
  Builds the decode DAG for the whole model and allocates the corresponding
  buffers.
- `python_vm.py`
  Reference implementation of the latency instructions in pure Python/Torch.
  This is the best semantic ground truth for understanding what each opcode is
  supposed to do.
- `mk.py`
  Marshals the latency globals into the bound `mk_llama` CUDA extension.

## Main instruction sequence per layer

The scheduler emits this logical flow per transformer block:

1. `LayerNorm_QKV_MatVecRopeAppend`
2. `PartialAttention`
3. `AttentionReduction`
4. `O_ProjResidual`
5. `LayerNormDoubleMatVecSiLU`
6. `DownProjResidual`

After the last layer it emits:

7. `RMS_LM_Head`

This is the entire decode-time forward path.

## Why this directory matters

If you want to understand the model flow without jumping across CUDA first,
`python_vm.py` is the easiest place to read the actual semantics end-to-end.

The CUDA files in `demos/low-latency-llama/` are optimized implementations of
the behaviors encoded here.

## Important design choices

### 1. Attention reduction is present but often skipped

The shipped latency schedule currently sets `skip_attn_reduction=True` and uses
`num_attention_partitions = 1` in `scheduler.py`.

That means the reduction op exists in the architecture, but the common decode
path usually bypasses the multi-partial reduction tree.

### 2. Barriers are semantic contracts

`python_vm.py` performs explicit barrier assertions before each op. Those checks
tell you exactly what data dependencies the CUDA side is expected to honor.

### 3. The scheduler is a model lowering pass

The scheduler is not just queue construction. It is effectively a compiler pass
from "Llama layer graph" into "instruction graph over flattened buffers."

## Dependency direction

This directory depends on:

- `megakernels/llama.py`
- `megakernels/instructions.py`
- `megakernels/python_vm.py`
- `megakernels/scheduler.py`

This directory is depended on by:

- `megakernels/dispatch.py`
- `megakernels/scripts/generate.py`
- `megakernels/scripts/llama_repl.py`

## Reading advice

Read the files in this order:

1. `instructions.py`
2. `scheduler.py`
3. `python_vm.py`
4. `mk.py`

Then cross over to `demos/low-latency-llama/CODESCOPE.md` for the CUDA side.
