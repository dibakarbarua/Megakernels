# include/ Code Scope

## Purpose

`include/` is the generic megakernel virtual machine runtime.

It does not know anything about Llama by itself. Instead, it provides:

- the persistent kernel skeleton
- instruction fetch and dispatch
- page allocation and page reuse
- shared-memory state layout
- worker-loop infrastructure
- timing hooks

The model-specific code in `demos/` plugs into this runtime by supplying:

- a `globals` type
- a set of op structs
- model-specific implementations of controller/loader/consumer/storer hooks

## Files and roles

- `megakernel.cuh`
  Top-level kernel body. Creates shared runtime state and launches the worker
  loops.
- `config.cuh`
  Runtime-wide configuration defaults: instruction width, timing width, number
  of consumer warps, page size, number of pages, and register budgets.
- `util.cuh`
  Core runtime state definitions and helper utilities. This is where the shared
  instruction ring, page tables, and event IDs live.
- `loader.cuh`
  Generic wrapper that dispatches the loader phase to the current opcode's
  loader hook.
- `storer.cuh`
  Generic wrapper for the storer phase.
- `launcher.cuh`
  Generic wrapper for the launcher phase.
- `consumer.cuh`
  Generic wrapper for consumer warps.
- `noop.cuh`
  Zero-op support so padded instruction slots are legal.
- `controller/`
  Subsystems used only by the controller warp.

## Runtime model

Each resident block owns a `state<config>` object from `util.cuh`.

That state contains:

- a ring of instruction slots
- one timing array per in-flight instruction slot
- dynamic semaphores for the active instruction
- logical-to-physical page mappings
- shared-memory pages used as the local working set

The block is partitioned into:

- 16 consumer warps
- 1 loader warp
- 1 storer warp
- 1 launcher warp
- 1 controller warp

This arrangement is defined in `config.cuh` and instantiated in
`megakernel.cuh`.

## Dependency direction

`include/` is depended on by:

- `demos/low-latency-llama/`
- any future model-specific megakernel demo

`include/` depends only on:

- ThunderKittens primitives
- CUDA compilation environment

## Reading advice

Read these files in order:

1. `config.cuh`
2. `util.cuh`
3. `megakernel.cuh`
4. `controller/CODESCOPE.md`

Then return to the model-specific code in `demos/low-latency-llama/`.
