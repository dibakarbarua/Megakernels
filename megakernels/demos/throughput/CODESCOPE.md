# throughput Mode Code Scope

## Purpose

This directory contains the higher-throughput, batched variant of the Python
control plane.

Compared with latency mode, it changes the instruction granularity:

- norms and attention decode are more explicitly separated
- matmul work is blocked across batch and output dimensions
- barriers track per-batch-block and per-output-block readiness

## Files and roles

- `instructions.py`
  Defines the throughput-mode globals and its instruction set.
- `scheduler.py`
  Builds the batched instruction stream for all layers plus LM head.
- `python_vm.py`
  Reference semantic implementation for the throughput opcodes.
- `mk.py`
  CUDA wrapper for the throughput megakernel ABI.

## Instruction flow

Per layer, the throughput path decomposes work into:

1. `PreAttnLayerNorm`
2. `QKV_MatMulRopeAppend`
3. `AttentionDecode`
4. `O_ProjResidual`
5. `PreMLP_Norm`
6. `GateSilu`
7. `UpMatMul`
8. `DownProjResidual`

Then after the final layer:

9. `PreLMHeadRMS`
10. `LM_Head`

This split makes the throughput path structurally different from the latency
path even though the underlying transformer semantics are the same.

## How to think about it

Latency mode tries to keep a single decode step as stitched as possible.

Throughput mode is more like a blocked batched dataflow graph. The scheduler
and barriers reason about:

- batch blocks
- output blocks
- intermediate blocks
- per-batch readiness rather than single-token readiness

## Caveats

- The throughput path appears less polished than the latency path.
- Some comments in the scheduler note broken or simplified dependencies around
  the LM head split.
- The corresponding CUDA kernel directory is not the one emphasized in the
  current repo README.

## Dependency direction

This directory depends on the same shared Python infrastructure as latency mode.

It is selected through `megakernels/dispatch.py` when `setting="throughput"`.
