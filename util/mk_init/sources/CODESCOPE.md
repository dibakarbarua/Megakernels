# util/mk_init/sources Code Scope

## Purpose

This directory is the template payload copied by `mk-init`.

It is a tiny starter megakernel project, not part of the main repo runtime.

## Files and roles

- `README.md`
  Generated-project getting-started instructions.
- `setup.py`
  Template build script that compiles a CUDA extension with `nvcc` and links
  against ThunderKittens and Megakernels headers.
- `src/`
  Template CUDA source files for the starter kernel.
- `tests/`
  Tiny starter validation script.

## Placeholder system

The scaffolder replaces:

- `{{PROJECT_NAME_LOWER}}`
- `{{PROJECT_NAME_UPPER}}`
- `{{PROJECT_NAME}}`

both inside file contents and in selected filenames.

## Dependency direction

Generated projects created from this template will depend on:

- ThunderKittens includes
- Megakernels includes
- pybind11
- a local CUDA toolchain

## Reading advice

Read `src/CODESCOPE.md` and `tests/CODESCOPE.md` to understand what gets
generated.
