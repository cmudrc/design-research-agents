from __future__ import annotations

from design_research_agents._contracts._llm import LLMMessage, LLMRequest, LLMResponse
from design_research_agents._contracts._workflow import ModelStep
from design_research_agents._runtime._workflow._executors._common import run_model_step


class _GenerateModelClient:
    def __init__(
        self,
        *,
        text: str = "model text",
    ) -> None:
        self.text = text
        self.requests: list[LLMRequest] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        model_name = request.model or "generated-default"
        return LLMResponse(model=model_name, text=self.text, provider="generate")


def _common_context() -> dict[str, object]:
    return {
        "prompt": "task",
        "dependency_results": {},
        "step_results": {},
    }


def _base_model_step(
    *,
    llm_client: object,
    response_schema: dict[str, object] | None = None,
) -> ModelStep:
    return ModelStep(
        step_id="model",
        llm_client=llm_client,  # type: ignore[arg-type]
        request_builder=lambda _ctx: LLMRequest(
            messages=[LLMMessage(role="user", content="evaluate")],
            model="model-x",
            response_schema=response_schema,
        ),
    )


def test_run_model_step_applies_output_schema_to_terminal_requests() -> None:
    generate_client = _GenerateModelClient(text='{"decision":"accept"}')
    output_schema = {
        "type": "object",
        "required": ["decision"],
        "properties": {"decision": {"type": "string"}},
    }
    step_context = _common_context()
    step_context["_workflow"] = {
        "is_terminal_step": True,
        "output_schema": output_schema,
    }

    success = run_model_step(
        step=_base_model_step(llm_client=generate_client),
        step_id="model",
        step_context=step_context,
    )

    assert success.success is True
    assert generate_client.requests[0].response_schema == output_schema


def test_run_model_step_preserves_explicit_response_schema_over_output_schema() -> None:
    generate_client = _GenerateModelClient(text='{"existing":"yes"}')
    explicit_schema = {
        "type": "object",
        "required": ["existing"],
        "properties": {"existing": {"type": "string"}},
    }
    step_context = _common_context()
    step_context["_workflow"] = {
        "is_terminal_step": True,
        "output_schema": {
            "type": "object",
            "required": ["decision"],
            "properties": {"decision": {"type": "string"}},
        },
    }

    success = run_model_step(
        step=_base_model_step(llm_client=generate_client, response_schema=explicit_schema),
        step_id="model",
        step_context=step_context,
    )

    assert success.success is True
    assert generate_client.requests[0].response_schema == explicit_schema


def test_run_model_step_does_not_apply_output_schema_when_non_terminal() -> None:
    generate_client = _GenerateModelClient(text="ok")
    step_context = _common_context()
    step_context["_workflow"] = {
        "is_terminal_step": False,
        "output_schema": {"type": "object"},
    }

    success = run_model_step(
        step=_base_model_step(llm_client=generate_client),
        step_id="model",
        step_context=step_context,
    )

    assert success.success is True
    assert generate_client.requests[0].response_schema is None
