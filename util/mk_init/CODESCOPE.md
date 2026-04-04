# util/mk_init Code Scope

## Purpose

This package implements the `mk-init` project scaffolding command.

It is analogous to a lightweight `npm init` for a new megakernel repo.

## Files and roles

- `main.py`
  CLI entrypoint. Prompts for a project name, creates the project directories,
  copies template files, and replaces placeholder tokens.
- `__init__.py`
  Minimal package marker.
- `sources/`
  Template tree copied into the generated project.

## Flow

`main.py` performs the following steps:

1. parse CLI args
2. determine the project name and target directory
3. validate the name
4. create `src/` and `tests/`
5. copy template files from `sources/`
6. replace `{{PROJECT_NAME...}}` placeholders
7. emit a starter `.gitignore`

## Dependency direction

This package depends on:

- Python stdlib only for the scaffolder itself
- files in `sources/`

It is packaged via `util/pyproject.toml`.

## Reading advice

Read `sources/CODESCOPE.md` next to understand what a generated starter project
looks like.
