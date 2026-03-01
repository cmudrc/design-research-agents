HTMLLLMClient
=============

``HTMLLLMClient`` is the built-in zero-dependency stand-in for quickstarts,
offline demos, CI smoke checks, and trace validation.

Highlights
----------

- No API key, network access, or provider SDK required
- Deterministic HTML output for the same prompt/messages
- Implements the normal ``LLMClient`` contract and tracing path
- Best suited for direct prompt-response flows, not structured output or tool-calling flows

Example
-------

.. code-block:: python

   from design_research_agents import DirectLLMCall, HTMLLLMClient

   with HTMLLLMClient() as llm_client:
       agent = DirectLLMCall(llm_client=llm_client)
       result = agent.run("Summarize the onboarding workflow in one paragraph.")
       print(result.output)

Behavior
--------

- Uses the most recent non-empty user message when available
- Falls back to concatenated message text when no user message is present
- Falls back to ``Hello from design-research-agents.`` when the request is empty
- Returns a simple HTML document containing the resolved text

Use this client when you want to exercise runtime contracts without binding the
base install to any commercial provider.
