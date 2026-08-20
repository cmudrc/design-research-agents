Installation
============

Requires Python 3.12+.

VS Code First
-------------

If you want a guided, editor-first path for creating a virtual environment,
installing the published package, and running a first script, start with
:doc:`vscode_setup`.

Package Install
---------------

Install the published package with a Python 3.12+ interpreter:

.. code-block:: bash

   python -m pip install design-research-agents

Windows note:
If ``python`` or ``pip`` resolve to an older interpreter, use
``py -3.12 -m pip install design-research-agents`` and
``py -3.12 -m venv .venv`` for virtual-environment setup.

Editable Install
----------------

.. code-block:: bash

   git clone https://github.com/cmudrc/design-research-agents.git
   cd design-research-agents
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e ".[dev]"

Maintainer Shortcut
-------------------

.. code-block:: bash

   make dev

Backend Extras
--------------

Backend clients are intentionally optional. Install only what your study needs.
For example:

.. code-block:: bash

   python -m pip install "design-research-agents[openai]"
   python -m pip install "design-research-agents[anthropic]"
   python -m pip install "design-research-agents[local]"
   python -m pip install "design-research-agents[full]"
   python -m pip install "design-research-agents[all]"

``full`` covers hosted + local backends. ``all`` adds the optional ChromaDB and
graph-memory extras on top of that backend bundle.

When working from a source checkout, replace ``design-research-agents`` with
``.`` and add ``-e`` to install the same extras in editable mode. Add ``dev``
only when you also need contributor tooling.

Use :doc:`dependencies_and_extras` for the complete matrix and platform caveats.
