from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from clawagents.config.config import EngineConfig
from clawagents.providers.llm import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
    NativeToolSchema,
    OnChunkCallback,
    create_provider,
)

logger = logging.getLogger(__name__)

RouteTier = Literal["simple", "coding", "reasoning"]


# Illustrative defaults only. Override these with your account's actual pricing
# if you want accurate USD accounting in production.
DEFAULT_PRICING_USD_PER_1M_TOKENS: dict[str, float] = {
    "gpt-5-nano": 0.10,
    "gpt-5-mini": 0.50,
    "gpt-5": 8.00,
    "gpt-5.3-codex": 5.00,
    "gpt-5.4": 15.00,
    "gpt-4o-mini": 0.30,
    "gpt-4o": 5.00,
    "gemini-3-flash": 0.35,
    "gemini-2.5-pro": 5.00,
    "claude-sonnet-4-5": 6.00,
}

_DISTRESS_PATTERNS: tuple[str, ...] = (
    "your last",
    "tool calls all failed",
    "stop and reconsider your approach",
    "fundamentally different strategy",
    "rethink #",
    "consecutive failures",
)

_CODING_PATTERNS: tuple[str, ...] = (
    "traceback",
    "stack trace",
    "error:",
    "exception",
    "failing test",
    "pytest",
    "syntaxerror",
    "typeerror",
    "nameerror",
    "assertionerror",
    "def ",
    "class ",
    "import ",
    "function",
    "method",
    "refactor",
    "debug",
    "fix the bug",
    "patch ",
    ".py",
    "src/",
)

_REASONING_PATTERNS: tuple[str, ...] = (
    "architecture",
    "tradeoff",
    "trade-off",
    "design",
    "migration",
    "migrate",
    "system design",
    "root cause",
    "deep reasoning",
    "proposal",
    "implementation plan",
    "compare approaches",
    "why does",
)


def _message_text(message: LLMMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    parts: list[str] = []
    for item in message.content:
        if isinstance(item, dict):
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif "text" in item:
                parts.append(str(item.get("text", "")))
    return "\n".join(parts)


@dataclass
class BudgetLedger:
    total_budget_usd: float
    pricing_usd_per_1m_tokens: dict[str, float] = field(default_factory=dict)
    spent_usd: float = 0.0
    tokens_consumed: dict[str, int] = field(default_factory=dict)
    spend_by_model: dict[str, float] = field(default_factory=dict)
    calls_by_model: dict[str, int] = field(default_factory=dict)
    unpriced_models: set[str] = field(default_factory=set)

    def remaining_budget_usd(self) -> float:
        return max(self.total_budget_usd - self.spent_usd, 0.0)

    def is_bankrupt(self) -> bool:
        return self.remaining_budget_usd() <= 0.0

    def _resolve_price(self, model: str) -> float:
        if model in self.pricing_usd_per_1m_tokens:
            return self.pricing_usd_per_1m_tokens[model]
        for key in sorted(self.pricing_usd_per_1m_tokens, key=len, reverse=True):
            if model.startswith(key):
                return self.pricing_usd_per_1m_tokens[key]
        self.unpriced_models.add(model)
        return 0.0

    def record_usage(self, model: str, tokens: int) -> float:
        safe_tokens = max(tokens, 0)
        rate = self._resolve_price(model)
        delta_usd = (safe_tokens / 1_000_000.0) * rate
        self.spent_usd += delta_usd
        self.tokens_consumed[model] = self.tokens_consumed.get(model, 0) + safe_tokens
        self.spend_by_model[model] = self.spend_by_model.get(model, 0.0) + delta_usd
        self.calls_by_model[model] = self.calls_by_model.get(model, 0) + 1
        return delta_usd

    def report(self) -> dict[str, Any]:
        models: dict[str, dict[str, Any]] = {}
        for model in sorted(self.tokens_consumed):
            models[model] = {
                "tokens_consumed": self.tokens_consumed[model],
                "calls": self.calls_by_model.get(model, 0),
                "spent_usd": round(self.spend_by_model.get(model, 0.0), 6),
                "usd_per_1m_tokens": self._resolve_price(model),
            }
        return {
            "total_budget_usd": round(self.total_budget_usd, 6),
            "spent_usd": round(self.spent_usd, 6),
            "remaining_budget_usd": round(self.remaining_budget_usd(), 6),
            "bankrupt": self.is_bankrupt(),
            "models": models,
            "unpriced_models": sorted(self.unpriced_models),
        }


@dataclass(frozen=True)
class RouteDecision:
    tier: RouteTier
    model: str
    reason: str


class EconomicSupervisorLLM(LLMProvider):
    name = "economic-supervisor"

    def __init__(
        self,
        config: EngineConfig,
        ledger: BudgetLedger,
        *,
        simple_model: str = "gpt-5-mini",
        coding_model: str | None = None,
        reasoning_model: str | None = None,
        pricing_usd_per_1m_tokens: dict[str, float] | None = None,
        route_window: int = 4,
        providers: dict[RouteTier, LLMProvider] | None = None,
        bankruptcy_message: str | None = None,
    ):
        self.ledger = ledger
        self.route_window = max(route_window, 1)
        self.bankruptcy_message = (
            bankruptcy_message
            or "Budget exhausted. Stop using tools and provide a concise summary of completed work, blockers, and the next best step."
        )

        if pricing_usd_per_1m_tokens:
            self.ledger.pricing_usd_per_1m_tokens.update(pricing_usd_per_1m_tokens)
        elif not self.ledger.pricing_usd_per_1m_tokens:
            self.ledger.pricing_usd_per_1m_tokens.update(DEFAULT_PRICING_USD_PER_1M_TOKENS)

        self.simple_model = simple_model
        self.coding_model = coding_model or config.openai_model or simple_model
        self.reasoning_model = reasoning_model or config.openai_model or self.coding_model

        if providers is not None:
            missing = {"simple", "coding", "reasoning"} - set(providers)
            if missing:
                raise ValueError(f"providers is missing route tiers: {sorted(missing)}")
            self._providers = providers
        else:
            self._providers = {
                "simple": self._build_provider(config, self.simple_model),
                "coding": self._build_provider(config, self.coding_model),
                "reasoning": self._build_provider(config, self.reasoning_model),
            }

        self._route_counts: dict[RouteTier, int] = {
            "simple": 0,
            "coding": 0,
            "reasoning": 0,
        }
        self._route_reason_counts: dict[str, int] = {}
        self._last_route: RouteDecision | None = None

    @staticmethod
    def _build_provider(config: EngineConfig, model_name: str) -> LLMProvider:
        cfg = config.model_copy(deep=True)
        return create_provider(model_name, cfg)

    def _recent_text(self, messages: list[LLMMessage]) -> str:
        recent = messages[-self.route_window:]
        return "\n".join(_message_text(message) for message in recent).lower()

    def _assess_complexity_and_route(self, messages: list[LLMMessage]) -> RouteDecision:
        recent_text = self._recent_text(messages)

        if any(pattern in recent_text for pattern in _DISTRESS_PATTERNS):
            return RouteDecision(
                tier="reasoning",
                model=self.reasoning_model,
                reason="distress-escalation",
            )

        reasoning_hits = sum(pattern in recent_text for pattern in _REASONING_PATTERNS)
        coding_hits = sum(pattern in recent_text for pattern in _CODING_PATTERNS)

        if reasoning_hits >= 1 and (reasoning_hits > coding_hits or "architecture" in recent_text or "implementation plan" in recent_text):
            return RouteDecision(
                tier="reasoning",
                model=self.reasoning_model,
                reason="deep-reasoning",
            )

        if coding_hits >= 1:
            return RouteDecision(
                tier="coding",
                model=self.coding_model,
                reason="coding-debugging",
            )

        return RouteDecision(
            tier="simple",
            model=self.simple_model,
            reason="low-complexity",
        )

    def get_financial_report(self) -> dict[str, Any]:
        report = self.ledger.report()
        report["routes"] = dict(self._route_counts)
        report["route_reasons"] = dict(sorted(self._route_reason_counts.items()))
        if self._last_route:
            report["last_route"] = {
                "tier": self._last_route.tier,
                "model": self._last_route.model,
                "reason": self._last_route.reason,
            }
        return report

    async def chat(
        self,
        messages: list[LLMMessage],
        on_chunk: OnChunkCallback = None,
        cancel_event: asyncio.Event | None = None,
        tools: list[NativeToolSchema] | None = None,
    ) -> LLMResponse:
        if self.ledger.is_bankrupt():
            return LLMResponse(
                content=self.bankruptcy_message,
                model=self.name,
                tokens_used=0,
            )

        decision = self._assess_complexity_and_route(messages)
        self._last_route = decision
        self._route_counts[decision.tier] += 1
        self._route_reason_counts[decision.reason] = self._route_reason_counts.get(decision.reason, 0) + 1

        provider = self._providers[decision.tier]
        response = await provider.chat(
            messages,
            on_chunk=on_chunk,
            cancel_event=cancel_event,
            tools=tools,
        )

        actual_model = response.model or decision.model
        delta_usd = self.ledger.record_usage(actual_model, response.tokens_used)
        logger.debug(
            "Economic supervisor routed %s -> %s (%s), tokens=%d, spend=$%.6f, remaining=$%.6f",
            decision.tier,
            actual_model,
            decision.reason,
            response.tokens_used,
            delta_usd,
            self.ledger.remaining_budget_usd(),
        )
        return response


class CostTrackingLLM(LLMProvider):
    """Wrap a single provider with ledger-based token and cost tracking."""

    name = "cost-tracking"

    def __init__(
        self,
        provider: LLMProvider,
        ledger: BudgetLedger,
        *,
        pricing_usd_per_1m_tokens: dict[str, float] | None = None,
        label: str = "fixed-model",
    ):
        self.provider = provider
        self.ledger = ledger
        self.label = label
        if pricing_usd_per_1m_tokens:
            self.ledger.pricing_usd_per_1m_tokens.update(pricing_usd_per_1m_tokens)
        elif not self.ledger.pricing_usd_per_1m_tokens:
            self.ledger.pricing_usd_per_1m_tokens.update(DEFAULT_PRICING_USD_PER_1M_TOKENS)

    async def chat(
        self,
        messages: list[LLMMessage],
        on_chunk: OnChunkCallback = None,
        cancel_event: asyncio.Event | None = None,
        tools: list[NativeToolSchema] | None = None,
    ) -> LLMResponse:
        response = await self.provider.chat(
            messages,
            on_chunk=on_chunk,
            cancel_event=cancel_event,
            tools=tools,
        )
        actual_model = response.model or self.label
        self.ledger.record_usage(actual_model, response.tokens_used)
        return response

    def get_financial_report(self) -> dict[str, Any]:
        report = self.ledger.report()
        report["routes"] = {"fixed": sum(report["models"][m]["calls"] for m in report["models"])}
        report["route_reasons"] = {"fixed-model": report["routes"]["fixed"]}
        return report
