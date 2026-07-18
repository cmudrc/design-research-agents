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

   python3 -m pip install design-research-agents

Windows note:
If ``python`` or ``pip`` resolve to an older interpreter, use
``py -3.12 -m pip install design-research-agents`` and
``py -3.12 -m venv .venv`` for virtual-environment setup.

Editable Install
----------------

.. code-block:: bash

   git clone https://github.com/cmudrc/design-research-agents.git
   cd design-research-agents
   python3 -m venv .venv
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

   pip install -e ".[openai]"
   pip install -e ".[anthropic]"
   pip install -e ".[local]"
   pip install -e ".[full]"
   pip install -e ".[all]"

``full`` covers hosted + local backends. ``all`` adds optional memory backends
and Gymnasium reference environments on top of that backend bundle.

Use :doc:`dependencies_and_extras` for the complete matrix and platform caveats.

Simulation Extras
-----------------

The core reinforcement-learning pattern has no Gymnasium dependency. Install the
``rl`` extra for the simulation-backed CartPole example:

.. code-block:: bash

   python -m pip install "design-research-agents[rl]"
