"""GeminiLLM's own logic, with only the network boundary mocked.

We have no API key and must not spend the user's free-tier quota to run the
test suite (the brief requires tests to run free and offline). What we can and
must verify without a network call is everything GeminiLLM does around that
call: prompt construction, JSON parsing, schema validation, and the
transient/permanent error mapping. Only ``genai.Client`` is faked.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from offboarding.domain.errors import PermanentToolError, TransientToolError
from offboarding.llm.fake import FakeLLM

EMPLOYEE = {
    "employee_id": "emp-001",
    "department": "Engineering",
    "last_day": "2026-09-30",
    "systems": ["github", "aws"],
    "has_company_hardware": True,
}

VALID_PLAN = {
    "items": [
        {"system": "github", "action": "Remove from orgs", "risk_level": "high"},
        {"system": "aws", "action": "Delete IAM user", "risk_level": "high"},
    ],
    "notes": "2 systems.",
}


class FakeInteractions:
    """Stands in for ``client.interactions``."""

    def __init__(self, *, output_text: str | None = None, raises: Exception | None = None):
        self._output_text = output_text
        self._raises = raises
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if self._raises:
            raise self._raises
        return SimpleNamespace(output_text=self._output_text)


class FakeClient:
    def __init__(self, interactions: FakeInteractions):
        self.interactions = interactions


@pytest.fixture
def gemini_module(monkeypatch):
    """Import GeminiLLM with google.genai.Client patched to a fake."""
    import google.genai as genai_pkg

    fake_interactions = FakeInteractions(output_text=json.dumps(VALID_PLAN))
    monkeypatch.setattr(
        genai_pkg, "Client", lambda **kw: FakeClient(fake_interactions)
    )
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")

    from offboarding.llm import gemini
    import importlib

    importlib.reload(gemini)
    return gemini, fake_interactions


class TestGeminiLLM:
    def test_missing_api_key_is_permanent(self, monkeypatch, gemini_module):
        gemini, _ = gemini_module
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(PermanentToolError, match="GEMINI_API_KEY"):
            gemini.GeminiLLM()

    def test_uses_the_low_cost_default_model(self, gemini_module):
        gemini, interactions = gemini_module
        llm = gemini.GeminiLLM()
        llm.generate_plan(EMPLOYEE)
        assert interactions.last_kwargs["model"] == "gemini-3.5-flash-lite"

    def test_disables_storage_for_pii(self, gemini_module):
        gemini, interactions = gemini_module
        gemini.GeminiLLM().generate_plan(EMPLOYEE)
        assert interactions.last_kwargs["store"] is False

    def test_requests_json_via_the_validated_schema(self, gemini_module):
        gemini, interactions = gemini_module
        gemini.GeminiLLM().generate_plan(EMPLOYEE)
        fmt = interactions.last_kwargs["response_format"]
        assert fmt["mime_type"] == "application/json"
        assert "items" in fmt["schema"]["properties"]

    def test_prompt_includes_the_employee_systems(self, gemini_module):
        gemini, interactions = gemini_module
        gemini.GeminiLLM().generate_plan(EMPLOYEE)
        assert "github" in interactions.last_kwargs["input"]
        assert "aws" in interactions.last_kwargs["input"]

    def test_valid_response_is_parsed_into_a_plan(self, gemini_module):
        gemini, _ = gemini_module
        plan = gemini.GeminiLLM().generate_plan(EMPLOYEE)
        assert {i.system for i in plan.items} == {"github", "aws"}

    def test_response_naming_an_unknown_system_is_permanent(
        self, monkeypatch, gemini_module
    ):
        gemini, interactions = gemini_module
        bad_plan = {
            "items": [
                {"system": "github", "action": "x", "risk_level": "high"},
                {"system": "aws", "action": "x", "risk_level": "high"},
                {"system": "salesforce", "action": "x", "risk_level": "low"},
            ],
            "notes": "",
        }
        interactions._output_text = json.dumps(bad_plan)
        with pytest.raises(PermanentToolError, match="does not have"):
            gemini.GeminiLLM().generate_plan(EMPLOYEE)

    def test_malformed_json_is_permanent_not_transient(
        self, gemini_module
    ):
        gemini, interactions = gemini_module
        interactions._output_text = "not valid json{{"
        with pytest.raises(PermanentToolError):
            gemini.GeminiLLM().generate_plan(EMPLOYEE)

    def test_transport_failure_is_transient(self, gemini_module):
        gemini, interactions = gemini_module
        interactions._raises = ConnectionError("boom")
        with pytest.raises(TransientToolError, match="ConnectionError"):
            gemini.GeminiLLM().generate_plan(EMPLOYEE)


class TestFactorySelection:
    def test_unknown_provider_name_is_permanent(self):
        from offboarding.llm.factory import build_llm

        with pytest.raises(PermanentToolError, match="unknown LLM provider"):
            build_llm("not-a-real-provider")

    def test_fake_is_the_default(self, monkeypatch):
        monkeypatch.delenv("OFFBOARDING_LLM_PROVIDER", raising=False)
        from offboarding.llm.factory import build_llm

        assert isinstance(build_llm(), FakeLLM)
