from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from clawagents.agent import create_claw_agent
from clawagents.benchmarking import BENCHMARK_TASKS
from clawagents.config.config import get_default_model, load_config
from clawagents.providers.economic import (
    DEFAULT_PRICING_USD_PER_1M_TOKENS,
    BudgetLedger,
    CostTrackingLLM,
    EconomicSupervisorLLM,
)
from clawagents.providers.llm import LLMProvider, create_provider
from clawagents.tools.exec import exec_tools
from clawagents.tools.filesystem import filesystem_tools


@dataclass
class TaskRun:
    task: str
    elapsed_s: float
    iterations: int
    tool_calls: int
    status: str
    spent_usd: float
    tokens_used: int
    report: dict[str, Any]


@dataclass
class ScenarioResult:
    name: str
    runs: list[TaskRun]
    financial_report: dict[str, Any]
    settings: dict[str, Any]


def _parse_price_overrides(items: list[str] | None) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Invalid --price value '{item}'. Expected model=usd_per_1m_tokens.")
        model, value = item.split("=", 1)
        result[model.strip()] = float(value.strip())
    return result


def _pricing_map(overrides: dict[str, float]) -> dict[str, float]:
    pricing = dict(DEFAULT_PRICING_USD_PER_1M_TOKENS)
    pricing.update(overrides)
    return pricing


def _report_totals(report: dict[str, Any]) -> tuple[float, int]:
    spent = float(report.get("spent_usd", 0.0))
    tokens = 0
    for model_data in report.get("models", {}).values():
        tokens += int(model_data.get("tokens_consumed", 0))
    return spent, tokens


def _report_delta(before: dict[str, Any], after: dict[str, Any]) -> tuple[float, int]:
    before_spent, before_tokens = _report_totals(before)
    after_spent, after_tokens = _report_totals(after)
    return after_spent - before_spent, after_tokens - before_tokens


def _merge_reports(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    if not left:
        return json.loads(json.dumps(right))
    merged = json.loads(json.dumps(left))
    merged["total_budget_usd"] = float(merged.get("total_budget_usd", 0.0)) + float(right.get("total_budget_usd", 0.0))
    merged["spent_usd"] = float(merged.get("spent_usd", 0.0)) + float(right.get("spent_usd", 0.0))
    merged["remaining_budget_usd"] = float(merged.get("remaining_budget_usd", 0.0)) + float(right.get("remaining_budget_usd", 0.0))
    merged["bankrupt"] = bool(merged.get("bankrupt", False) or right.get("bankrupt", False))

    merged_models = merged.setdefault("models", {})
    for model, info in right.get("models", {}).items():
        entry = merged_models.setdefault(model, {
            "tokens_consumed": 0,
            "calls": 0,
            "spent_usd": 0.0,
            "usd_per_1m_tokens": info.get("usd_per_1m_tokens", 0.0),
        })
        entry["tokens_consumed"] += int(info.get("tokens_consumed", 0))
        entry["calls"] += int(info.get("calls", 0))
        entry["spent_usd"] = round(float(entry.get("spent_usd", 0.0)) + float(info.get("spent_usd", 0.0)), 6)
        entry["usd_per_1m_tokens"] = info.get("usd_per_1m_tokens", entry.get("usd_per_1m_tokens", 0.0))

    merged_routes = merged.setdefault("routes", {})
    for key, value in right.get("routes", {}).items():
        merged_routes[key] = int(merged_routes.get(key, 0)) + int(value)

    merged_reasons = merged.setdefault("route_reasons", {})
    for key, value in right.get("route_reasons", {}).items():
        merged_reasons[key] = int(merged_reasons.get(key, 0)) + int(value)

    merged_unpriced = set(merged.get("unpriced_models", []))
    merged_unpriced.update(right.get("unpriced_models", []))
    merged["unpriced_models"] = sorted(merged_unpriced)

    if right.get("last_route"):
        merged["last_route"] = right["last_route"]
    return merged


def _snapshot(provider: Any) -> dict[str, Any]:
    getter = getattr(provider, "get_financial_report", None)
    if callable(getter):
        return getter()
    return {}


def _clone_config() -> Any:
    return load_config().model_copy(deep=True)


def _create_fixed_model_provider(model_name: str, pricing: dict[str, float]) -> CostTrackingLLM:
    config = _clone_config()
    provider = create_provider(model_name, config)
    return CostTrackingLLM(
        provider=provider,
        ledger=BudgetLedger(total_budget_usd=1_000_000.0),
        pricing_usd_per_1m_tokens=pricing,
        label=model_name,
    )


def _create_supervisor_provider(
    *,
    simple_model: str,
    coding_model: str,
    reasoning_model: str,
    pricing: dict[str, float],
) -> EconomicSupervisorLLM:
    config = _clone_config()
    return EconomicSupervisorLLM(
        config,
        BudgetLedger(total_budget_usd=1_000_000.0),
        simple_model=simple_model,
        coding_model=coding_model,
        reasoning_model=reasoning_model,
        pricing_usd_per_1m_tokens=pricing,
    )


def _create_agent(
    provider: LLMProvider,
    *,
    streaming: bool,
    rethink: bool,
    trajectory: bool,
    max_iterations: int,
):
    return create_claw_agent(
        model=provider,
        tools=filesystem_tools + exec_tools,
        streaming=streaming,
        rethink=rethink,
        trajectory=trajectory,
        max_iterations=max_iterations,
    )


async def _run_tasks_for_provider(
    provider: Any,
    tasks: list[str],
    *,
    streaming: bool,
    rethink: bool,
    trajectory: bool,
    max_iterations: int,
) -> list[TaskRun]:
    agent = _create_agent(
        provider,
        streaming=streaming,
        rethink=rethink,
        trajectory=trajectory,
        max_iterations=max_iterations,
    )
    runs: list[TaskRun] = []
    for task in tasks:
        before = _snapshot(provider)
        t0 = time.time()
        result = await agent.invoke(task, max_iterations=max_iterations)
        elapsed = time.time() - t0
        after = result.financial_report or _snapshot(provider)
        spent_delta, token_delta = _report_delta(before, after)
        runs.append(TaskRun(
            task=task,
            elapsed_s=elapsed,
            iterations=result.iterations,
            tool_calls=result.tool_calls,
            status=result.status,
            spent_usd=spent_delta,
            tokens_used=token_delta,
            report=after,
        ))
    return runs


async def _run_scenario(
    *,
    name: str,
    provider_factory,
    tasks: list[str],
    streaming: bool,
    rethink: bool,
    trajectory: bool,
    max_iterations: int,
    repeats: int,
    settings: dict[str, Any],
) -> ScenarioResult:
    all_runs: list[TaskRun] = []
    final_report: dict[str, Any] = {}

    for _ in range(repeats):
        provider = provider_factory()
        runs = await _run_tasks_for_provider(
            provider,
            tasks,
            streaming=streaming,
            rethink=rethink,
            trajectory=trajectory,
            max_iterations=max_iterations,
        )
        all_runs.extend(runs)
        final_report = _merge_reports(final_report, _snapshot(provider))

    return ScenarioResult(
        name=name,
        runs=all_runs,
        financial_report=final_report,
        settings=settings,
    )


def _group_by_task(runs: list[TaskRun]) -> dict[str, list[TaskRun]]:
    grouped: dict[str, list[TaskRun]] = {}
    for run in runs:
        grouped.setdefault(run.task, []).append(run)
    return grouped


def _avg(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _pct_delta(new: float, old: float) -> float:
    if old == 0:
        return 0.0
    return ((new - old) / old) * 100.0


def _build_summary(fixed: ScenarioResult, supervisor: ScenarioResult) -> dict[str, Any]:
    fixed_grouped = _group_by_task(fixed.runs)
    supervisor_grouped = _group_by_task(supervisor.runs)

    tasks_summary: list[dict[str, Any]] = []
    for task in fixed_grouped:
        if task not in supervisor_grouped:
            continue
        fixed_runs = fixed_grouped[task]
        sup_runs = supervisor_grouped[task]
        fixed_cost = _avg([run.spent_usd for run in fixed_runs])
        sup_cost = _avg([run.spent_usd for run in sup_runs])
        fixed_tokens = _avg([float(run.tokens_used) for run in fixed_runs])
        sup_tokens = _avg([float(run.tokens_used) for run in sup_runs])
        fixed_time = _avg([run.elapsed_s for run in fixed_runs])
        sup_time = _avg([run.elapsed_s for run in sup_runs])
        tasks_summary.append({
            "task": task,
            "fixed_model": {
                "avg_elapsed_s": round(fixed_time, 3),
                "avg_spent_usd": round(fixed_cost, 6),
                "avg_tokens_used": round(fixed_tokens, 1),
            },
            "economic_supervisor": {
                "avg_elapsed_s": round(sup_time, 3),
                "avg_spent_usd": round(sup_cost, 6),
                "avg_tokens_used": round(sup_tokens, 1),
            },
            "delta": {
                "spent_usd": round(sup_cost - fixed_cost, 6),
                "spent_pct": round(_pct_delta(sup_cost, fixed_cost), 2),
                "tokens_used": round(sup_tokens - fixed_tokens, 1),
                "tokens_pct": round(_pct_delta(sup_tokens, fixed_tokens), 2),
                "elapsed_s": round(sup_time - fixed_time, 3),
                "elapsed_pct": round(_pct_delta(sup_time, fixed_time), 2),
            },
        })

    fixed_total_spend, fixed_total_tokens = _report_totals(fixed.financial_report)
    sup_total_spend, sup_total_tokens = _report_totals(supervisor.financial_report)

    return {
        "fixed_model": {
            "settings": fixed.settings,
            "total_spent_usd": round(fixed_total_spend, 6),
            "total_tokens_used": fixed_total_tokens,
            "financial_report": fixed.financial_report,
        },
        "economic_supervisor": {
            "settings": supervisor.settings,
            "total_spent_usd": round(sup_total_spend, 6),
            "total_tokens_used": sup_total_tokens,
            "financial_report": supervisor.financial_report,
        },
        "comparison": {
            "spent_usd": round(sup_total_spend - fixed_total_spend, 6),
            "spent_pct": round(_pct_delta(sup_total_spend, fixed_total_spend), 2),
            "tokens_used": sup_total_tokens - fixed_total_tokens,
            "tokens_pct": round(_pct_delta(float(sup_total_tokens), float(fixed_total_tokens)), 2),
        },
        "tasks": tasks_summary,
    }


def _print_summary(summary: dict[str, Any]) -> None:
    fixed = summary["fixed_model"]
    supervisor = summary["economic_supervisor"]
    comparison = summary["comparison"]

    print("Scenario totals")
    print(
        f"  Fixed model:          ${fixed['total_spent_usd']:.6f} | "
        f"{fixed['total_tokens_used']} tokens"
    )
    print(
        f"  Economic supervisor:  ${supervisor['total_spent_usd']:.6f} | "
        f"{supervisor['total_tokens_used']} tokens"
    )
    print(
        f"  Delta:                ${comparison['spent_usd']:+.6f} "
        f"({comparison['spent_pct']:+.2f}%) | "
        f"{comparison['tokens_used']:+d} tokens ({comparison['tokens_pct']:+.2f}%)"
    )
    print("")
    print("| Task | Fixed Cost | Supervisor Cost | Cost Delta % | Fixed Tokens | Supervisor Tokens | Token Delta % | Fixed Time (s) | Supervisor Time (s) |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in summary["tasks"]:
        print(
            f"| {row['task']} | "
            f"{row['fixed_model']['avg_spent_usd']:.6f} | "
            f"{row['economic_supervisor']['avg_spent_usd']:.6f} | "
            f"{row['delta']['spent_pct']:+.2f}% | "
            f"{row['fixed_model']['avg_tokens_used']:.1f} | "
            f"{row['economic_supervisor']['avg_tokens_used']:.1f} | "
            f"{row['delta']['tokens_pct']:+.2f}% | "
            f"{row['fixed_model']['avg_elapsed_s']:.3f} | "
            f"{row['economic_supervisor']['avg_elapsed_s']:.3f} |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark a fixed single-model workflow against EconomicSupervisorLLM."
    )
    parser.add_argument("--fixed-model", help="Single model for the non-supervisor baseline. Default: active model from config.")
    parser.add_argument("--simple-model", default="gpt-5-mini")
    parser.add_argument("--coding-model", help="Supervisor coding model. Default: fixed model.")
    parser.add_argument("--reasoning-model", help="Supervisor reasoning model. Default: fixed model.")
    parser.add_argument("--task", action="append", dest="tasks", help="Benchmark task. Can be passed multiple times.")
    parser.add_argument("--price", action="append", dest="prices", help="Override pricing as model=usd_per_1m_tokens.")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument("--no-rethink", action="store_true")
    parser.add_argument("--trajectory", action="store_true")
    parser.add_argument("--output", default="benchmark_economic_supervisor_report.json")
    args = parser.parse_args()

    config = load_config()
    fixed_model = args.fixed_model or get_default_model(config)
    coding_model = args.coding_model or fixed_model
    reasoning_model = args.reasoning_model or fixed_model
    tasks = list(args.tasks or BENCHMARK_TASKS)
    pricing = _pricing_map(_parse_price_overrides(args.prices))

    fixed_result = asyncio.run(_run_scenario(
        name="fixed-model",
        provider_factory=lambda: _create_fixed_model_provider(fixed_model, pricing),
        tasks=tasks,
        streaming=not args.no_streaming,
        rethink=not args.no_rethink,
        trajectory=args.trajectory,
        max_iterations=args.max_iterations,
        repeats=max(args.repeats, 1),
        settings={
            "model": fixed_model,
            "streaming": not args.no_streaming,
            "rethink": not args.no_rethink,
            "trajectory": args.trajectory,
            "repeats": max(args.repeats, 1),
        },
    ))
    supervisor_result = asyncio.run(_run_scenario(
        name="economic-supervisor",
        provider_factory=lambda: _create_supervisor_provider(
            simple_model=args.simple_model,
            coding_model=coding_model,
            reasoning_model=reasoning_model,
            pricing=pricing,
        ),
        tasks=tasks,
        streaming=not args.no_streaming,
        rethink=not args.no_rethink,
        trajectory=args.trajectory,
        max_iterations=args.max_iterations,
        repeats=max(args.repeats, 1),
        settings={
            "simple_model": args.simple_model,
            "coding_model": coding_model,
            "reasoning_model": reasoning_model,
            "streaming": not args.no_streaming,
            "rethink": not args.no_rethink,
            "trajectory": args.trajectory,
            "repeats": max(args.repeats, 1),
        },
    ))

    summary = _build_summary(fixed_result, supervisor_result)
    output_path = Path(args.output)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _print_summary(summary)
    print(f"\nSaved JSON report to {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
