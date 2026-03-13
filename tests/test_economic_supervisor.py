import pytest

from clawagents.agent import ClawAgent
from clawagents.config.config import EngineConfig
from clawagents.providers.economic import BudgetLedger, EconomicSupervisorLLM
from clawagents.providers.llm import LLMMessage, LLMProvider, LLMResponse
from clawagents.tools.registry import ToolRegistry


class StubProvider(LLMProvider):
    name = "stub"

    def __init__(self, model: str, content: str = "done", tokens_used: int = 100):
        self.model = model
        self.content = content
        self.tokens_used = tokens_used
        self.calls = 0

    async def chat(self, messages, on_chunk=None, cancel_event=None, tools=None) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            content=self.content,
            model=self.model,
            tokens_used=self.tokens_used,
        )


def _config() -> EngineConfig:
    return EngineConfig(
        openai_api_key="test-key",
        openai_model="gpt-5",
    )


def test_budget_ledger_tracks_spend_and_unpriced_models():
    ledger = BudgetLedger(
        total_budget_usd=1.00,
        pricing_usd_per_1m_tokens={"gpt-5": 2.0},
    )

    spend = ledger.record_usage("gpt-5.4", 500_000)
    missing = ledger.record_usage("unknown-model", 1000)
    report = ledger.report()

    assert spend == pytest.approx(1.0)
    assert missing == 0.0
    assert report["bankrupt"] is True
    assert report["models"]["gpt-5.4"]["tokens_consumed"] == 500_000
    assert report["unpriced_models"] == ["unknown-model"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "expected_tier"),
    [
        ("List the Python files under src and summarize them.", "simple"),
        ("Traceback: fix this failing pytest in src/app.py.", "coding"),
        ("Stop and reconsider your approach before trying again.", "reasoning"),
    ],
)
async def test_supervisor_routes_by_recent_context(message: str, expected_tier: str):
    simple = StubProvider("gpt-5-mini", tokens_used=100)
    coding = StubProvider("gpt-5.3-codex", tokens_used=200)
    reasoning = StubProvider("gpt-5.4", tokens_used=300)
    supervisor = EconomicSupervisorLLM(
        _config(),
        BudgetLedger(total_budget_usd=10.0),
        simple_model="gpt-5-mini",
        coding_model="gpt-5.3-codex",
        reasoning_model="gpt-5.4",
        providers={
            "simple": simple,
            "coding": coding,
            "reasoning": reasoning,
        },
        pricing_usd_per_1m_tokens={
            "gpt-5-mini": 1.0,
            "gpt-5.3-codex": 2.0,
            "gpt-5.4": 3.0,
        },
    )

    response = await supervisor.chat([LLMMessage(role="user", content=message)])
    report = supervisor.get_financial_report()

    assert response.model == {
        "simple": "gpt-5-mini",
        "coding": "gpt-5.3-codex",
        "reasoning": "gpt-5.4",
    }[expected_tier]
    assert report["routes"][expected_tier] == 1
    assert simple.calls == (1 if expected_tier == "simple" else 0)
    assert coding.calls == (1 if expected_tier == "coding" else 0)
    assert reasoning.calls == (1 if expected_tier == "reasoning" else 0)


@pytest.mark.asyncio
async def test_bankruptcy_short_circuits_and_financial_report_is_exposed():
    simple = StubProvider("gpt-5-mini")
    coding = StubProvider("gpt-5.3-codex")
    reasoning = StubProvider("gpt-5.4")
    supervisor = EconomicSupervisorLLM(
        _config(),
        BudgetLedger(total_budget_usd=0.0),
        simple_model="gpt-5-mini",
        coding_model="gpt-5.3-codex",
        reasoning_model="gpt-5.4",
        providers={
            "simple": simple,
            "coding": coding,
            "reasoning": reasoning,
        },
    )
    agent = ClawAgent(
        llm=supervisor,
        tools=ToolRegistry(),
        streaming=False,
    )

    state = await agent.invoke("List the Python files.")

    assert "Budget exhausted" in state.result
    assert state.financial_report is not None
    assert state.financial_report["bankrupt"] is True
    assert simple.calls == 0
    assert coding.calls == 0
    assert reasoning.calls == 0
