# megakernels/scripts Code Scope

## Purpose

This directory contains the user-facing entrypoints.

These scripts answer practical questions such as:

- how do I run generation?We i
- how do I compare PyTorch vs VM vs megakernel?
- how do I launch a small REPL?
- how do I profile the reference model?
- how do I benchmark against server-style engines?

## Files and roles

- `generate.py`
  Main driver. Loads a model, runs prefill, builds a schedule, and
  executes generation under one of several backends.
- `llama_repl.py`
  Interactive chat loop around the megakernel or PyTorch backend.
- `diff_test.py`
  Comparison harness for validating outputs between execution modes.
- `make_torch_profile.py`
  Helper script for generating a PyTorch execution profile.
- `bench_engines.py`
  Benchmarks external serving engines by launching a server and driving requests.

## Most important script

`generate.py` is the best top-level entrypoint for understanding the whole flow.

It clearly shows the order:

1. load tokenizer
2. load model
3. tokenize prompt
4. run PyTorch prefill
5. build schedule
6. choose backend
7. generate decode tokens

## Dependency direction

These scripts depend on:

- `megakernels/dispatch.py`
- `megakernels/generators.py`
- `megakernels/llama.py`
- `megakernels/scheduler.py`

They are the topmost layer of the first-party runtime stack.

## Reading advice

Read `generate.py` first, then `generators.py`, then the selected mode's
directory under `megakernels/demos/`.
