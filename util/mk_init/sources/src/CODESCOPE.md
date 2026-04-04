# util/mk_init/sources/src Code Scope

## Purpose

This directory contains the starter CUDA source files for a generated project.

The template is intentionally small. Its job is to give you a minimal starting
point for defining:

- a project-specific config
- a project-specific CUDA extension entrypoint

## Files

- `config.cuh`
  Starter configuration header for the generated project.
- `{{PROJECT_NAME_LOWER}}.cu`
  Template CUDA entrypoint module for the generated project.

## How it relates to the main repo

These files are not used by the Megakernels runtime directly.

They show the expected shape of a new project that wants to build on:

- Megakernels runtime headers
- ThunderKittens primitives
- pybind11 kernel binding

## Porting relevance

If you want to create a new model-specific megakernel outside this repo, this
template is the shortest path from blank directory to compilable extension.
