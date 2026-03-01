from __future__ import annotations

import importlib
import sys


def test_importing_llm_module_does_not_eagerly_import_openai_clients() -> None:
    module_names = (
        "design_research_agents.llm",
        "design_research_agents.llm.clients",
        "design_research_agents.llm.clients._openai_service",
        "design_research_agents.llm.clients._azure_openai_service",
    )
    saved_modules = {name: sys.modules.get(name) for name in module_names}
    try:
        for name in module_names:
            sys.modules.pop(name, None)

        llm_module = importlib.import_module("design_research_agents.llm")

        assert "OpenAIServiceLLMClient" in llm_module.__all__
        assert "design_research_agents.llm.clients._openai_service" not in sys.modules
        assert "design_research_agents.llm.clients._azure_openai_service" not in sys.modules
    finally:
        for name in module_names:
            sys.modules.pop(name, None)
        for name, module in saved_modules.items():
            if module is not None:
                sys.modules[name] = module


def test_importing_model_selector_does_not_eagerly_import_openai_clients() -> None:
    module_names = (
        "design_research_agents._model_selection._selector",
        "design_research_agents.llm",
        "design_research_agents.llm.clients",
        "design_research_agents.llm.clients._openai_service",
    )
    saved_modules = {name: sys.modules.get(name) for name in module_names}
    try:
        for name in module_names:
            sys.modules.pop(name, None)

        selector_module = importlib.import_module("design_research_agents._model_selection._selector")

        assert "OpenAIServiceLLMClient" in selector_module._CLIENT_CLASS_REFS
        assert "design_research_agents.llm.clients._openai_service" not in sys.modules
    finally:
        for name in module_names:
            sys.modules.pop(name, None)
        for name, module in saved_modules.items():
            if module is not None:
                sys.modules[name] = module


def test_importing_llm_clients_module_does_not_eagerly_import_openai_clients() -> None:
    module_names = (
        "design_research_agents.llm.clients",
        "design_research_agents.llm.clients._openai_service",
        "design_research_agents.llm.clients._azure_openai_service",
    )
    saved_modules = {name: sys.modules.get(name) for name in module_names}
    try:
        for name in module_names:
            sys.modules.pop(name, None)

        clients_module = importlib.import_module("design_research_agents.llm.clients")

        assert "OpenAIServiceLLMClient" in clients_module.__all__
        assert "design_research_agents.llm.clients._openai_service" not in sys.modules
        assert "design_research_agents.llm.clients._azure_openai_service" not in sys.modules
    finally:
        for name in module_names:
            sys.modules.pop(name, None)
        for name, module in saved_modules.items():
            if module is not None:
                sys.modules[name] = module
