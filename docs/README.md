# Documentation Maintenance

## Build Docs Locally

- `make docs-check`
- `make docs-build`

## Example Page Generation

Example pages are generated from runnable scripts via `scripts/generate_example_docs.py`.
Update script docstrings/comments, then run docs generation checks before commit.

## Docstring Style

Use Google-style docstrings for public APIs and examples where applicable.
Run `make docstrings-check` before merge.

## Page-Writing Conventions

- Keep the homepage short: title, tagline, concise framing, quickstart callout, section-oriented links, and only the minimum ecosystem/contribution notes needed for orientation.
- Keep the root hidden home-page toctree section-first so the PyData header and sidebar stay stable.
- Emphasize reproducible agent studies, explicit runtime boundaries, and interpretable execution traces.
- Use public API imports in snippets unless an internal boundary is being explained intentionally.

## Table vs Prose Rule

Prefer compact tables for scanning. Preserve nuance in narrative paragraphs directly below the table. Do not use tables to carry long explanatory sentences.

## Cross-links

Use `:doc:` for internal pages and explicit links for sibling repositories when needed.

## Branding

- The ecosystem figure is the source of truth for package colors.
- This repo's canonical docs brand color is `#DF5127`.
- Keep docs CSS tokens, `drc-light.png`, `drc-dark.png`, and `favicon.ico` aligned when updating docs styling.

## API Page Updates

When top-level exports change, update:

- `docs/api.rst`
- homepage/API references
- relevant quickstart/examples snippets
