# util/mk_init/sources/tests Code Scope

## Purpose

This directory contains the starter test payload copied into a generated
project.

## Files

- `test_example.py`
  Minimal smoke test that allocates instruction and timing buffers on CUDA and
  calls the generated example kernel.

## Why it exists

The test is intentionally simple. It verifies that:

- the extension can be imported
- the generated kernel symbol is callable
- the most basic VM-style buffer contract is wired up

## Relationship to the main repo

This is template code only. It does not participate in the Megakernels runtime
inside this repository.
