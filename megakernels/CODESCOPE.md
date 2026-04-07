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

NOTES: 
- This is pretty much the exact PyTorch implementation of the model in PyTorch.
- There is a wrapper object that instantiates the Llama model (layers+LMHead)
  - Then there are additional utilities that flattens model parameters
- the setup_caches() function creates a stacked KV cache per layer
- the stack_params() is the key flattening function as shown

```
self_attns = [x.self_attn for x in layers]
        mlps = [x.mlp for x in layers]

        o_projs = [x.o_proj for x in self_attns]
        self_attn_lns = [x.input_layernorm for x in self_attns]

        mlp_lns = [x.input_layernorm for x in mlps]
        up_projs = [x.up_proj for x in mlps]
        gate_projs = [x.gate_proj for x in mlps]
        down_projs = [x.down_proj for x in mlps]

        stacked_o_proj = stack_and_reassign(o_projs, "weight")
        stacked_self_attn_ln_weights = stack_and_reassign(self_attn_lns, "weight")
        stacked_mlp_ln_weights = stack_and_reassign(mlp_lns, "weight")
        stacked_up_proj = stack_and_reassign(up_projs, "weight")
        stacked_gate_proj = stack_and_reassign(gate_projs, "weight")
        stacked_down_proj = stack_and_reassign(down_projs, "weight")

        qkv_weights = []
        for self_attn in self_attns:
            cat_weight = torch.cat(
                [
                    self_attn.q_proj.weight,
                    self_attn.k_proj.weight,
                    self_attn.v_proj.weight,
                ],
                dim=0,
            )
            qkv_weights.append(cat_weight)

        stacked_qkv_weights = torch.stack(qkv_weights, dim=0)

        for i, self_attn in enumerate(self_attns):
            qkv_weight = stacked_qkv_weights[i]
            q_weight, k_weight, v_weight = qkv_weight.split(
                [
                    self.config.num_attention_heads * self.config.head_dim,
                    self.config.num_key_value_heads * self.config.head_dim,
                    self.config.num_key_value_heads * self.config.head_dim,
                ],
                dim=0,
            )

            self_attn.q_proj.weight[:] = q_weight
            self_attn.k_proj.weight[:] = k_weight
            self_attn.v_proj.weight[:] = v_weight

        self.stacked_params = StackedParams(
            qkv_proj=stacked_qkv_weights,
            o_proj=stacked_o_proj,
            attn_ln_weight=stacked_self_attn_ln_weights,
            mlp_ln_weight=stacked_mlp_ln_weights,
            up_proj=stacked_up_proj,
            gate_proj=stacked_gate_proj,
            down_proj=stacked_down_proj,
        )

```

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

KEY CONCEPTS:
- Instruction
  - opcode, prev_opcode, serialize(returns full 32-int instruction encoding)
  - The definition of an instruction is usually a single op / fused ops for a given layer.
- DAG
  - Each operation is a DAG node in a graph
  - The operation will likely depend on prior operations completing
    - Example earlier operations in its layer, all ops in previous layers.
  - Once we distill the entire model in a DAG, we encode the computations
- Schedule
  - A schedule consists of a statically pre-scheduled sequence of instructions on each SM
  - The flattened model is represented as a DAG (above) of instructions and dependencies
  - All SMs are initially available to schedule work
  - The Scheduler creates a heap of SMs and Instructions
    - The SM heap returns SM that will be available soonest
    - The instruction heap returns the longest "ready" instruction
  - Finally we create the static schedule by going "instruction-by-instruction"
    - We map each instruction to an SM and add it to its "instruction queue"
- ScheduleBuiler
  - wrapper class to return Schedule
- SM Allocations (TBD)
  - Different assignments to load balance and avoid SM quantization
  
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

## Encoding each sequence

```
class BatchState:
    input_ids: Tensor
    position_ids: Tensor | None = None
    seq_len: int | None = None
    output_ids: Tensor | None = None
    hidden_states: Tensor | None = None
    position_embeddings: tuple[Tensor, Tensor] | None = None

    kv_indices: Tensor | None = None
    kv_indptr: Tensor | None = None
    kv_last_page_lens: Tensor | None = None
    kv_seqlens: Tensor | None = None
    qo_indptr: Tensor | None = None
    prefill_wrapper: Any | None = None
    decode_wrapper: Any | None = None

    def __post_init__(self):
        if self.seq_len is None:
            self.seq_len = self.input_ids.shape[1]
```

## Reading advice

Read these next:

1. `scripts/CODESCOPE.md`
2. `demos/CODESCOPE.md`

Then choose:

- `demos/latency/CODESCOPE.md` for the main shipped path
- `demos/throughput/CODESCOPE.md` for the batched variant
