Run An Example In VS Code
=========================

Use this page when you want to try ``design-research-agents`` in VS Code.
Choose the installed-package path for a first user workflow, or the source
checkout path when you want to run the repository's checked-in examples and
development checks.

The checked-in ``examples/`` directory lives in the repository source. Do not
assume those files are present inside the PyPI wheel.

Requirements
------------

- Python 3.12 or newer.
- VS Code with the Python extension.
- A VS Code integrated terminal.

For the editor setup itself, follow the official `Getting Started with Python in
VS Code <https://code.visualstudio.com/docs/python/python-tutorial/>`_
tutorial. It covers installing the Python and Pylance extensions, opening a
folder, selecting an interpreter, and running a Python file in VS Code.

Installed Package From PyPI
---------------------------

Open an empty folder in VS Code, then create and activate a virtual
environment from ``Terminal > New Terminal``.

On macOS or Linux:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install design-research-agents

On Windows PowerShell:

.. code-block:: powershell

   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install design-research-agents

Run ``Python: Select Interpreter`` from the command palette and choose the
interpreter inside ``.venv``. If VS Code does not list it, enter the interpreter
path manually:

- macOS/Linux: ``.venv/bin/python``
- Windows: ``.venv\Scripts\python.exe``

Create ``hello_agents.py`` in the workspace folder:

.. code-block:: python

   import json

   import design_research_agents as drag


   class HelloWorldLLMClient:
       def generate(self, request: drag.LLMRequest) -> drag.LLMResponse:
           del request
           return drag.LLMResponse(
               text="Hello from design-research-agents.",
               model="local-demo",
               provider="local-demo",
           )

       def default_model(self) -> str:
           return "local-demo"


   agent = drag.DirectLLMCall(llm_client=HelloWorldLLMClient())
   result = agent.run(
       prompt="Say hello to a new design research teammate.",
       request_id="example-vscode-hello-world-001",
   )
   print(json.dumps(result.summary(), ensure_ascii=True, indent=2, sort_keys=True))

This example uses only the published package API and does not require an API key
or model server.

Run the file with VS Code's ``Run Python File`` action, or run:

.. code-block:: bash

   python hello_agents.py

A successful run prints a JSON summary in the integrated terminal:

.. code-block:: json

   {
     "error": null,
     "final_output": "Hello from design-research-agents.",
     "success": true,
     "terminated_reason": null,
     "trace": {
       "request_id": "example-vscode-hello-world-001"
     }
   }

Source Checkout For Repository Examples
---------------------------------------

Use this path when you want the checked-in examples, docs, tests, and optional
development tooling.

.. code-block:: bash

   git clone https://github.com/cmudrc/design-research-agents.git
   cd design-research-agents
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip setuptools wheel
   python -m pip install -e ".[dev]"

Equivalent maintainer shortcut:

.. code-block:: bash

   make dev

Run the deterministic VS Code onboarding example from the integrated terminal:

.. code-block:: bash

   python examples/agents/vscode_hello_world.py
   make examples-smoke

First Development Checks
------------------------

Run the checks from VS Code's integrated terminal:

.. code-block:: bash

   make test
   make qa
   make docs-check

``make qa`` runs linting, formatting checks, type checks, and tests. Run
``make coverage`` before merge when changing tested behavior.

Optional Runtimes
-----------------

The base install supports deterministic local examples. Install model-client
or simulation extras only when a workflow needs them. For example:

.. code-block:: bash

   python -m pip install "design-research-agents[openai]"
   python -m pip install "design-research-agents[rl]"

Use :doc:`dependencies_and_extras` for the full extras list.

Troubleshooting
---------------

- If VS Code imports fail but the terminal works, reselect the ``.venv``
  interpreter and reload the window.
- If ``make`` uses the wrong Python, activate ``.venv`` in the terminal or run
  ``PYTHON=.venv/bin/python make test``.
- If Windows activation is blocked, switch the terminal profile to Command
  Prompt and run ``.\.venv\Scripts\activate.bat``.
- If a model backend import is missing, install the matching backend extra
  rather than broad optional dependencies.
- Avoid committing generated runtime output under ``artifacts/``,
  ``docs/_build/``, or local virtual environment directories.
