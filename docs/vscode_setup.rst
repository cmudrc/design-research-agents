VS Code Setup Guide
===================

This guide covers the library-specific steps for using the published
``design-research-agents`` package in VS Code.

1. Install The Prerequisites
----------------------------

Install VS Code, and make sure you have access to a Python 3.12 or newer
interpreter.

- `Visual Studio Code <https://code.visualstudio.com/download>`_
- `Python 3.12 or newer <https://www.python.org/downloads/>`_ if you do not
  already have a compatible interpreter

For the editor setup itself, follow the official `Getting Started with Python in
VS Code <https://code.visualstudio.com/docs/python/python-tutorial/>`_
tutorial. It covers installing the Python and Pylance extensions, opening a
folder, selecting an interpreter, and running a Python file in VS Code.

After that, come back here for the package-specific steps below.

Windows note:
When installing Python, enable the option that adds ``python`` to your
``PATH`` so the integrated terminal can find it.

2. Create Or Open A Workspace Folder
------------------------------------

In VS Code, choose ``File > Open Folder...`` and open a folder for your own
project, such as ``design-research-study``.

You do not need to clone this repository to use the library.

3. Create A Virtual Environment
-------------------------------

Open ``Terminal > New Terminal`` and create a virtual environment in your
workspace folder.

On macOS or Linux:

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate

On Windows:

.. code-block:: powershell

   python -m venv .venv
   .\.venv\Scripts\activate

4. Install The Package
----------------------

With the virtual environment active, run:

.. code-block:: bash

   python -m pip install --upgrade pip
   pip install design-research-agents

If you want to connect to a specific model backend later, install only the
extra you need. For example:

.. code-block:: bash

   pip install "design-research-agents[openai]"

Use :doc:`dependencies_and_extras` for the full extras list.

5. Select The Python Interpreter
--------------------------------

VS Code usually detects ``.venv`` automatically. If it does not:

1. Open the Command Palette.
2. Run ``Python: Select Interpreter``.
3. Choose the interpreter inside this folder's ``.venv``.

6. Create A First Script
------------------------

Create a file named ``hello_agents.py`` in your workspace with this example:

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
   result = agent.run("Say hello to a new design research teammate.")
   print(json.dumps(result.summary(), ensure_ascii=True, indent=2, sort_keys=True))

This example uses only the published package API and does not require an API key
or model server.

7. Run The Script
-----------------

You can run the file in either of these ways:

- Click ``Run Python File`` in the editor.
- Press ``F5`` and choose ``Python Debugger: Current File`` if VS Code asks.

A successful run prints a JSON summary in the integrated terminal.

8. Troubleshooting
------------------

If the terminal says ``python`` or ``python3`` cannot be found:

- confirm Python finished installing
- restart VS Code after installation
- on Windows, reinstall Python and enable the PATH option

If imports are underlined after installation:

1. Run ``Python: Select Interpreter``.
2. Pick the interpreter inside ``.venv``.

If you want a terminal-first workflow instead of VS Code:

- use :doc:`installation`
- then continue with :doc:`quickstart`
