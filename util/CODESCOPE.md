# util/ Code Scope

## Purpose

`util/` is not part of the runtime path for serving or generation.

Instead, it packages developer tooling for creating a new megakernel project
from templates.

## Files and roles

- `pyproject.toml`
  Packages the `mk-init` tool as a small installable Python utility.
- `mk_init/`
  The actual scaffolding package and template sources.

## Why this directory matters

If you want to clone the architecture of this repo for another model or
accelerator experiment, `mk_init/` shows how the authors expect a new project to
be bootstrapped.

## Dependency direction

This directory is mostly independent from the main runtime. It depends on:

- standard Python packaging
- template files in `mk_init/sources/`

It is not involved in the decode-time execution flow.

## Reading advice

Read `mk_init/CODESCOPE.md` next if you want the project-generation path.
