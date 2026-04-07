# Megakernels Code Scope

## What this repository is

Megakernels is split into two cooperating halves:

1. A Python control plane that loads an LLM, flattens its layer weights into
   megakernel-friendly tensors, lowers the model into an instruction DAG, and
   serializes one instruction queue per SM.
2. A CUDA execution plane that launches one resident block per SM and runs a
   small interpreter. That interpreter fetches the precomputed instructions and
   executes the corresponding fused operator implementations.

The repo currently centers on a low-latency Llama decode demo. Prefill still
runs through ordinary PyTorch. The megakernel path takes over for decode-time
single-token forwards.

## Read this repo in this order

1. `README.md`
2. `megakernels/CODESCOPE.md`
3. `megakernels/demos/latency/CODESCOPE.md`
4. `demos/low-latency-llama/CODESCOPE.md`
5. `include/CODESCOPE.md`

If you want the higher-throughput batched variant after that, continue with:

1. `megakernels/demos/throughput/CODESCOPE.md`

If you want to understand how to start a new megakernel project, read:

1. `util/CODESCOPE.md`

## Top-level directory map

- `megakernels/`
  Python package. This is the control plane and reference implementation.
- `demos/`
  Model-specific CUDA demo kernels. The shipped one is `low-latency-llama`.
- `include/`
  Generic megakernel VM runtime: controller, page allocator, worker loops,
  instruction dispatch, and shared-memory state.
- `util/`
  Project scaffolding and templates for creating new megakernel projects.
- `ThunderKittens/`
  External submodule dependency. Not documented here because it is vendored
  upstream rather than authored in this repo.

## End-to-end flow

### 1. Entry scripts choose a mode and load the model

The main entrypoints are in `megakernels/scripts/`. The most direct one is
`megakernels/scripts/generate.py`.

That script:

- loads a tokenizer
- loads `LlamaForCausalLM` from Hugging Face weights
- runs a normal PyTorch prefill
- builds a reusable schedule
- chooses an execution engine: pure PyTorch, Python VM, or CUDA megakernel

### 2. Python lowers the model into instructions

The schedule builders in `megakernels/demos/{latency,throughput}/scheduler.py`
translate the model into a sequence or DAG of instruction objects.

Those instruction objects describe logical work such as:

TODO: Map out the computations in each layer and the sizes of weights and activations

- pre-attention RMS norm
- QKV projection with RoPE and KV append
- attention decode
- O projection with residual
- MLP gate and up projections
- down projection with residual
- LM head

### 3. Python serializes one queue per SM

`megakernels/scheduler.py` pads each instruction to 32 integers, arranges the
result as `instructions[sm, queue_index, 32]`, and allocates a parallel timing
buffer.

This schedule is reused across token-generation steps as long as the model
shape and scheduling strategy do not change.

### 4. The runtime writes the current token embedding and position

At decode time the generator:

- embeds the current input token with normal PyTorch
- copies the hidden state into the global activation buffer
- sets `pos_id`
- clears the barrier counters
- calls the bound CUDA kernel

### 5. The megakernel interpreter executes the whole forward

The CUDA side launches one resident block per SM. Each block contains:

- consumer warps that do math
- a loader warp
- a storer warp
- a launcher warp
- a controller warp

The controller warp fetches that SM's next instruction, constructs the needed
semaphores, computes page reuse, and releases the other warps to execute the
instruction.

### 6. Python reads logits and picks the next token

After the megakernel finishes, Python reads the global logits buffer, applies
`argmax`, writes the next token into the output token array, and repeats.

## Important design assumptions

- The low-latency demo is optimized for decode, not prompt prefill.
- The main CUDA demo is hardcoded for a specific Llama shape in
  `demos/low-latency-llama/llama.cuh`.
- The VM runtime in `include/` is generic, but every new model family still
  needs a model-specific globals struct, op set, and scheduler lowering logic.
- Porting to another accelerator means preserving the architecture:
  instruction IR, explicit dependencies, local-memory page allocation, and
  persistent workers.

## Intentional exclusions

This documentation does not add `CODESCOPE.md` files under:

- `.git/` because it is repository metadata
- `ThunderKittens/` because it is an external submodule with its own internal
  structure and ownership
