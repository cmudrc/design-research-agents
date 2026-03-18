VS Code Setup Guide
===================

This guide is the easiest path for new contributors and non-technical
researchers. It walks from zero to a working VS Code workspace without assuming
shell fluency.

What You Get
------------

After following this page, you will have:

- recommended VS Code extensions installed
- a local ``.venv`` created for this repository
- an editable development install of ``design-research-agents``
- a working ``F5`` launch target for a no-network hello-world example
- ready-to-use VS Code tasks for setup, tests, and docs

1. Install The Prerequisites
----------------------------

Install these three tools first:

- `Visual Studio Code <https://code.visualstudio.com/download>`_
- `Python 3.12 or newer <https://www.python.org/downloads/>`_
- `Git <https://git-scm.com/downloads>`_

Windows note:
When installing Python, enable the option that adds ``python`` to your
``PATH``. That makes the VS Code setup task work without extra manual steps.

2. Clone And Open The Repository
--------------------------------

If you prefer a fully visual flow, clone the repository from inside VS Code:

1. Open VS Code.
2. Open the Command Palette with ``Ctrl+Shift+P`` or ``Cmd+Shift+P``.
3. Run ``Git: Clone``.
4. Paste ``https://github.com/cmudrc/design-research-agents.git``.
5. Choose a local folder.
6. Click ``Open`` when VS Code offers to open the cloned repository.

If you already cloned the repository another way, use ``File > Open Folder`` and
open the project root.

3. Install The Recommended Extensions
-------------------------------------

When the repository opens, VS Code should prompt you to install the recommended
extensions from ``.vscode/extensions.json``.

Accept that prompt. The workspace recommends:

- Python
- Pylance
- Ruff

If you dismiss the prompt by accident, open the Extensions panel and install the
recommended extensions manually.

4. Run The Setup Task
---------------------

This repository includes a VS Code task named ``Setup Project``.

To run it:

1. Open ``Terminal > Run Task...``.
2. Choose ``Setup Project``.
3. Wait for the task to finish.

The task will:

- create ``.venv`` if it does not already exist
- upgrade ``pip`` inside that environment
- install the project in editable mode with development dependencies

The first run may take a few minutes. Re-running the task is safe after pulling
new changes or switching branches.

5. Confirm The Python Interpreter
---------------------------------

VS Code usually detects ``.venv`` automatically after setup. If it does not:

1. Open the Command Palette.
2. Run ``Python: Select Interpreter``.
3. Choose the interpreter inside this repository's ``.venv``.

The workspace settings in ``.vscode/settings.json`` also tell VS Code to:

- treat ``src`` as an analysis path for imports
- auto-activate the environment in integrated terminals
- use ``pytest`` for test discovery

6. Run The First Example With F5
--------------------------------

This repository includes a dedicated onboarding launch configuration in
``.vscode/launch.json``.

To use it:

1. Press ``F5``.
2. Choose ``VS Code: Hello World Example`` if VS Code asks which launch target to run.

That launch target runs ``examples/agents/vscode_hello_world.py``. It uses a
small local stub client, so it does not require an external model server or API
key. A successful run prints a JSON summary to the integrated terminal.

7. Common VS Code Workflows
---------------------------

Once setup is complete, these built-in tasks are the most useful starting
points:

- ``Setup Project``: refresh the editable install after branch changes
- ``Run Tests``: run the pytest suite with the local virtual environment
- ``Build Docs``: regenerate example docs and build the Sphinx site locally

You can run them anytime from ``Terminal > Run Task...``.

8. Troubleshooting
------------------

If the setup task says ``python`` or ``python3`` cannot be found:

- confirm Python finished installing
- restart VS Code after installation
- on Windows, reinstall Python and enable the PATH option

If imports are underlined even after setup:

1. rerun ``Setup Project``
2. run ``Python: Select Interpreter``
3. pick the interpreter inside ``.venv``

If debugging fails after switching branches:

- rerun ``Setup Project`` so the editable install matches the checked-out code

If you prefer a terminal-first workflow instead of VS Code:

- use :doc:`installation`
- then continue with :doc:`quickstart`
