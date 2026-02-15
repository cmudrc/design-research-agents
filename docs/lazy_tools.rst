Lazy Tools
==========

Lazy tools are local ``.py`` or ``.sh`` scripts discovered from configured
search paths. Each script declares tool metadata in a header docblock/comment
block within the first 120 lines.

Required directives:

- ``@tool_name``
- ``@description``
- ``@inputs``
- ``@outputs`` (must include ``stdout_json: true``)
- ``@capabilities``

CLI helpers:

.. code-block:: bash

   dra lazy lint examples/lazy_tools
   dra lazy list --config tool_runtime.yaml
   dra lazy run lazy::rubric_score --json '{"text":"hello"}' --config tool_runtime.yaml

See examples in:

- ``examples/lazy_tools/python/rubric_score.py``
- ``examples/lazy_tools/bash/repo_quickscan.sh``
