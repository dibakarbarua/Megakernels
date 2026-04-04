# include/controller Code Scope

## Purpose

This directory contains the controller-warp subsystems.

The controller warp is the only part of the runtime that understands the life
cycle of an instruction slot:

1. wait for the old slot to retire
2. fetch the next serialized instruction
3. compute logical-to-physical page reuse
4. construct the semaphores needed by the opcode
5. release the worker warps to execute it

## Files and roles

- `controller.cuh`
  Main controller loop. This is the control heart of the interpreter.
- `instruction_fetch.cuh`
  Copies one serialized instruction from global memory into the current shared
  instruction slot.
- `page_allocator.cuh`
  Asks the active opcode how logical pages should be remapped for the next
  instruction.
- `semaphore_constructor.cuh`
  Asks the active opcode how many semaphores it needs and initializes them.
- `timings_store.cuh`
  Writes timing data for completed instruction slots.

## Key idea

The controller warp does not execute model math. It manages resource reuse and
instruction turnover so the math warps can stay resident and keep working.

That separation is the main reason the runtime can overlap work without paying
kernel-launch overhead between transformer sub-ops.

## Dependency direction

This directory depends on:

- `include/util.cuh`
- opcode-specific `controller` hooks from the model demo

This directory is depended on by:

- `include/megakernel.cuh`

## What to understand here for porting

If you port the runtime to another backend, this directory is where you would
recreate:

- instruction fetch
- resource-table updates
- in-flight instruction-slot reuse
- event and timing bookkeeping

If you only port to another model family on CUDA, this directory often stays
almost unchanged.
