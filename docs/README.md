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

- Keep homepages in this order: title, tagline, what it does, highlights, typical workflow, ecosystem integration, start here.
- Prefer concise academic prose and direct explanation of research relevance.
- Use public API imports in snippets unless an internal boundary is being explained intentionally.

## Table vs Prose Rule

Prefer compact tables for scanning. Preserve nuance in narrative paragraphs directly below the table. Do not use tables to carry long explanatory sentences.

## Cross-links

Use `:doc:` for internal pages and explicit links for sibling repositories when needed.

## API Page Updates

When top-level exports change, update:

- `docs/api.rst`
- homepage/API references
- relevant quickstart/examples snippets
