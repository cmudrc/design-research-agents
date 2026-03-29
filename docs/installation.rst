Installation
============

VS Code First
-------------

If you want a guided, editor-first path for creating a virtual environment,
installing the published package, and running a first script, start with
:doc:`vscode_setup`.

Package Install
---------------

.. code-block:: bash

   pip install design-research-agents

Editable Install
----------------

.. code-block:: bash

   git clone https://github.com/cmudrc/design-research-agents.git
   cd design-research-agents
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -e ".[dev]"

Maintainer Shortcut
-------------------

.. code-block:: bash

   make dev

Backend Extras
--------------

Backend clients are intentionally optional. Install only what your study needs.
For example:

.. code-block:: bash

   pip install -e ".[openai]"
   pip install -e ".[anthropic]"
   pip install -e ".[local]"
   pip install -e ".[full]"

Use :doc:`dependencies_and_extras` for the complete matrix and platform caveats.
