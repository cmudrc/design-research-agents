Documentation Automation
========================

This repository treats documentation surfaces as a checked contract. The same
baseline is intended to be portable across the design-research module repos
where the utilities apply, with repo-specific exceptions called out explicitly.

Shared Baseline
---------------

- ``README.md``, the PyPI long description, and :doc:`index` share the same five-badge baseline: CI, Coverage, Examples Passing, Public API In Examples, and Docs.
- ``make docs-check``, ``make docs-build``, and ``make docs-linkcheck`` are the canonical docs validation commands.
- ``scripts/generate_example_docs.py`` is the canonical path for generating example pages from runnable sources.
- ``make examples-metrics`` is the canonical badge/metrics refresh path for deterministic example reporting.
- ``.github/workflows/docs-pages.yml`` owns docs checks, strict docs builds, and Pages deployment.
- ``.github/workflows/ci.yml`` owns the main CI gate plus coverage and examples badge refresh on trusted ``main`` pushes.
- ``.github/workflows/examples.yml`` owns the full deterministic examples suite.
- ``.github/workflows/update-release-readme.yml`` runs ``scripts/update_release_readme.py`` to keep the monthly release callout in ``README.md`` current.

Copyable Snippets
-----------------

Docs builds include ``sphinx_copybutton`` with prompt stripping enabled so shell
and Python blocks copy cleanly without leading ``$`` or ``>>>`` prompts.

Repo-Specific Exception
-----------------------

This repo keeps the example-oriented badge/report pair
(``Examples Passing`` and ``Public API In Examples``) because runnable examples
are part of the public-surface contract here. Sibling repos can omit those
surfaces when they do not publish comparable example metrics.

Regression Checks
-----------------

``scripts/check_docs_consistency.py`` verifies the badge baseline, copybutton
wiring, the checked docs-automation hooks, and the newcomer-facing
:doc:`where_to_start` guide.
