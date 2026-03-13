from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import statistics
import sys
import tempfile
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
from clawagents.sandbox.local import LocalBackend


@dataclass
class BenchmarkTaskSpec:
    task_id: str
    display_name: str
    prompt: str
    task_type: str
    grading_type: str | None = None
    automated_checks: str | None = None
    workspace_files: list[dict[str, Any]] | None = None


@dataclass
class TaskRun:
    task_id: str
    task_name: str
    elapsed_s: float
    iterations: int
    tool_calls: int
    status: str
    spent_usd: float
    tokens_used: int
    score: float | None
    grading_type: str | None
    breakdown: dict[str, float]
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
    sandbox: Any,
    streaming: bool,
    rethink: bool,
    trajectory: bool,
    max_iterations: int,
):
    return create_claw_agent(
        model=provider,
        sandbox=sandbox,
        streaming=streaming,
        rethink=rethink,
        trajectory=trajectory,
        max_iterations=max_iterations,
    )


async def _run_tasks_for_provider(
    provider: Any,
    tasks: list[BenchmarkTaskSpec],
    *,
    skill_dir: Path | None,
    streaming: bool,
    rethink: bool,
    trajectory: bool,
    max_iterations: int,
) -> list[TaskRun]:
    runs: list[TaskRun] = []
    for task in tasks:
        workspace = _prepare_workspace(task, skill_dir)
        try:
            agent = _create_agent(
                provider,
                sandbox=LocalBackend(str(workspace)),
                streaming=streaming,
                rethink=rethink,
                trajectory=trajectory,
                max_iterations=max_iterations,
            )
            before = _snapshot(provider)
            t0 = time.time()
            result = await agent.invoke(task.prompt, max_iterations=max_iterations)
            elapsed = time.time() - t0
            after = result.financial_report or _snapshot(provider)
            spent_delta, token_delta = _report_delta(before, after)
            score, breakdown = _grade_run(
                task,
                messages=getattr(result, "messages", []),
                workspace=workspace,
                status=result.status,
            )
            runs.append(TaskRun(
                task_id=task.task_id,
                task_name=task.display_name,
                elapsed_s=elapsed,
                iterations=result.iterations,
                tool_calls=result.tool_calls,
                status=result.status,
                spent_usd=spent_delta,
                tokens_used=token_delta,
                score=score,
                grading_type=task.grading_type,
                breakdown=breakdown,
                report=after,
            ))
        finally:
            if task.task_type == "skill-main":
                shutil.rmtree(workspace, ignore_errors=True)
    return runs


async def _run_scenario(
    *,
    name: str,
    provider_factory,
    tasks: list[BenchmarkTaskSpec],
    skill_dir: Path | None,
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
            skill_dir=skill_dir,
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
        grouped.setdefault(run.task_id, []).append(run)
    return grouped


def _avg(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _pct_delta(new: float, old: float) -> float:
    if old == 0:
        return 0.0
    return ((new - old) / old) * 100.0


def _score_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "mean": round(_avg(values), 4),
        "std": round(_std(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def _total_requests(report: dict[str, Any]) -> int:
    return sum(int(model_data.get("calls", 0)) for model_data in report.get("models", {}).values())


def _efficiency_summary(
    *,
    scores: list[float],
    total_tokens: int,
    total_cost_usd: float,
    total_execution_time_s: float,
    task_count: int,
    report: dict[str, Any],
) -> dict[str, Any]:
    total_score = sum(scores)
    return {
        "score_stats": _score_stats(scores),
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost_usd, 6),
        "total_requests": _total_requests(report),
        "total_execution_time_seconds": round(total_execution_time_s, 3),
        "tokens_per_task": round(total_tokens / task_count, 1) if task_count > 0 else 0.0,
        "cost_per_task_usd": round(total_cost_usd / task_count, 6) if task_count > 0 else 0.0,
        "score_per_1k_tokens": round(total_score / (total_tokens / 1000.0), 6) if total_tokens > 0 else None,
        "score_per_dollar": round(total_score / total_cost_usd, 4) if total_cost_usd > 0 else None,
    }


def _build_summary(fixed: ScenarioResult, supervisor: ScenarioResult) -> dict[str, Any]:
    fixed_grouped = _group_by_task(fixed.runs)
    supervisor_grouped = _group_by_task(supervisor.runs)

    tasks_summary: list[dict[str, Any]] = []
    for task in fixed_grouped:
        if task not in supervisor_grouped:
            continue
        fixed_runs = fixed_grouped[task]
        sup_runs = supervisor_grouped[task]
        task_name = fixed_runs[0].task_name
        fixed_cost = _avg([run.spent_usd for run in fixed_runs])
        sup_cost = _avg([run.spent_usd for run in sup_runs])
        fixed_tokens = _avg([float(run.tokens_used) for run in fixed_runs])
        sup_tokens = _avg([float(run.tokens_used) for run in sup_runs])
        fixed_time = _avg([run.elapsed_s for run in fixed_runs])
        sup_time = _avg([run.elapsed_s for run in sup_runs])
        fixed_scores = [run.score for run in fixed_runs if run.score is not None]
        sup_scores = [run.score for run in sup_runs if run.score is not None]
        fixed_mean_score = _avg(fixed_scores) if fixed_scores else None
        sup_mean_score = _avg(sup_scores) if sup_scores else None
        tasks_summary.append({
            "task_id": task,
            "task_name": task_name,
            "fixed_model": {
                "avg_elapsed_s": round(fixed_time, 3),
                "avg_spent_usd": round(fixed_cost, 6),
                "avg_tokens_used": round(fixed_tokens, 1),
                "score_stats": _score_stats(fixed_scores),
                "graded_runs": len(fixed_scores),
                "tokens_per_score_point": round(sum(run.tokens_used for run in fixed_runs) / sum(fixed_scores), 1) if fixed_scores and sum(fixed_scores) > 0 else None,
            },
            "economic_supervisor": {
                "avg_elapsed_s": round(sup_time, 3),
                "avg_spent_usd": round(sup_cost, 6),
                "avg_tokens_used": round(sup_tokens, 1),
                "score_stats": _score_stats(sup_scores),
                "graded_runs": len(sup_scores),
                "tokens_per_score_point": round(sum(run.tokens_used for run in sup_runs) / sum(sup_scores), 1) if sup_scores and sum(sup_scores) > 0 else None,
            },
            "delta": {
                "spent_usd": round(sup_cost - fixed_cost, 6),
                "spent_pct": round(_pct_delta(sup_cost, fixed_cost), 2),
                "tokens_used": round(sup_tokens - fixed_tokens, 1),
                "tokens_pct": round(_pct_delta(sup_tokens, fixed_tokens), 2),
                "elapsed_s": round(sup_time - fixed_time, 3),
                "elapsed_pct": round(_pct_delta(sup_time, fixed_time), 2),
                "mean_score": round((sup_mean_score or 0.0) - (fixed_mean_score or 0.0), 4) if fixed_scores or sup_scores else None,
            },
        })

    fixed_total_spend, fixed_total_tokens = _report_totals(fixed.financial_report)
    sup_total_spend, sup_total_tokens = _report_totals(supervisor.financial_report)
    fixed_total_scores = [run.score for run in fixed.runs if run.score is not None]
    sup_total_scores = [run.score for run in supervisor.runs if run.score is not None]
    fixed_total_time = sum(run.elapsed_s for run in fixed.runs)
    sup_total_time = sum(run.elapsed_s for run in supervisor.runs)
    fixed_efficiency = _efficiency_summary(
        scores=fixed_total_scores,
        total_tokens=fixed_total_tokens,
        total_cost_usd=fixed_total_spend,
        total_execution_time_s=fixed_total_time,
        task_count=len(fixed_grouped),
        report=fixed.financial_report,
    )
    sup_efficiency = _efficiency_summary(
        scores=sup_total_scores,
        total_tokens=sup_total_tokens,
        total_cost_usd=sup_total_spend,
        total_execution_time_s=sup_total_time,
        task_count=len(supervisor_grouped),
        report=supervisor.financial_report,
    )

    return {
        "fixed_model": {
            "settings": fixed.settings,
            **fixed_efficiency,
            "graded_runs": len(fixed_total_scores),
            "financial_report": fixed.financial_report,
        },
        "economic_supervisor": {
            "settings": supervisor.settings,
            **sup_efficiency,
            "graded_runs": len(sup_total_scores),
            "financial_report": supervisor.financial_report,
        },
        "comparison": {
            "spent_usd": round(sup_total_spend - fixed_total_spend, 6),
            "spent_pct": round(_pct_delta(sup_total_spend, fixed_total_spend), 2),
            "tokens_used": sup_total_tokens - fixed_total_tokens,
            "tokens_pct": round(_pct_delta(float(sup_total_tokens), float(fixed_total_tokens)), 2),
            "mean_score": round(
                ((sup_efficiency["score_stats"]["mean"] or 0.0) - (fixed_efficiency["score_stats"]["mean"] or 0.0)),
                4,
            ) if fixed_total_scores or sup_total_scores else None,
            "score_per_1k_tokens": round(
                ((sup_efficiency["score_per_1k_tokens"] or 0.0) - (fixed_efficiency["score_per_1k_tokens"] or 0.0)),
                6,
            ) if fixed_efficiency["score_per_1k_tokens"] is not None and sup_efficiency["score_per_1k_tokens"] is not None else None,
            "score_per_dollar": round(
                ((sup_efficiency["score_per_dollar"] or 0.0) - (fixed_efficiency["score_per_dollar"] or 0.0)),
                4,
            ) if fixed_efficiency["score_per_dollar"] is not None and sup_efficiency["score_per_dollar"] is not None else None,
        },
        "tasks": tasks_summary,
    }


def _print_summary(summary: dict[str, Any]) -> None:
    fixed = summary["fixed_model"]
    supervisor = summary["economic_supervisor"]
    comparison = summary["comparison"]

    print("Scenario totals")
    print(
        f"  Fixed model:          ${fixed['total_cost_usd']:.6f} | "
        f"{fixed['total_tokens']} tokens | "
        f"mean score={_format_metric(fixed['score_stats'].get('mean'))} | "
        f"score/1K tok={_format_metric(fixed.get('score_per_1k_tokens'))} | "
        f"score/$={_format_metric(fixed.get('score_per_dollar'))}"
    )
    print(
        f"  Economic supervisor:  ${supervisor['total_cost_usd']:.6f} | "
        f"{supervisor['total_tokens']} tokens | "
        f"mean score={_format_metric(supervisor['score_stats'].get('mean'))} | "
        f"score/1K tok={_format_metric(supervisor.get('score_per_1k_tokens'))} | "
        f"score/$={_format_metric(supervisor.get('score_per_dollar'))}"
    )
    print(
        f"  Delta:                ${comparison['spent_usd']:+.6f} "
        f"({comparison['spent_pct']:+.2f}%) | "
        f"{comparison['tokens_used']:+d} tokens ({comparison['tokens_pct']:+.2f}%) | "
        f"mean score={_format_delta(comparison.get('mean_score'))} | "
        f"score/1K tok={_format_delta(comparison.get('score_per_1k_tokens'), precision=6)} | "
        f"score/$={_format_delta(comparison.get('score_per_dollar'), precision=4)}"
    )
    print("")
    print("| Task | Fixed Mean | Fixed Std | Supervisor Mean | Supervisor Std | Fixed Cost | Supervisor Cost | Cost Delta % |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in summary["tasks"]:
        print(
            f"| {row['task_id']} | "
            f"{_format_metric(row['fixed_model']['score_stats'].get('mean'))} | "
            f"{_format_metric(row['fixed_model']['score_stats'].get('std'))} | "
            f"{_format_metric(row['economic_supervisor']['score_stats'].get('mean'))} | "
            f"{_format_metric(row['economic_supervisor']['score_stats'].get('std'))} | "
            f"{row['fixed_model']['avg_spent_usd']:.6f} | "
            f"{row['economic_supervisor']['avg_spent_usd']:.6f} | "
            f"{row['delta']['spent_pct']:+.2f}% |"
        )


def _format_metric(value: float | None, *, pct: bool = False) -> str:
    if value is None:
        return "n/a"
    if pct:
        return f"{value * 100:.1f}%"
    return f"{value:.3f}"


def _format_delta(value: float | None, *, pct: bool = False, precision: int = 3) -> str:
    if value is None:
        return "n/a"
    if pct:
        return f"{value * 100:+.1f}%"
    return f"{value:+.{precision}f}"


def _format_currency(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:.6f}"


def _extract_automated_grader(automated_checks: str | None):
    if not automated_checks:
        return None
    match = re.search(r"```python\s*(.*?)\s*```", automated_checks, re.DOTALL)
    if not match:
        return None
    namespace: dict[str, Any] = {}
    exec(match.group(1), namespace)
    grade_func = namespace.get("grade")
    return grade_func if callable(grade_func) else None


def _grade_scores(scores: dict[str, Any]) -> tuple[float, dict[str, float]]:
    normalized: dict[str, float] = {}
    for key, value in scores.items():
        try:
            normalized[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return (_avg(list(normalized.values())) if normalized else 0.0, normalized)


def _parse_tool_calls(content: str) -> list[dict[str, Any]]:
    text = content.strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []

    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return []

    tool_calls: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        tool_name = item.get("tool")
        params = item.get("args", {})
        if isinstance(tool_name, str):
            tool_calls.append({
                "type": "toolCall",
                "name": tool_name,
                "params": params if isinstance(params, dict) else {},
            })
    return tool_calls


def _normalize_tool_call(tool_name: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    normalized_params = dict(params)
    if tool_name == "read_file" and "path" in normalized_params and "files" not in normalized_params:
        normalized_params["files"] = [normalized_params["path"]]

    variants = [{
        "type": "toolCall",
        "name": tool_name,
        "params": normalized_params,
    }]

    alias_map = {
        "execute": "execute_command",
    }
    alias = alias_map.get(tool_name)
    if alias:
        variants.append({
            "type": "toolCall",
            "name": alias,
            "params": normalized_params,
        })
    return variants


def _messages_to_transcript(messages: list[Any]) -> list[dict[str, Any]]:
    transcript: list[dict[str, Any]] = []
    for msg in messages:
        role = getattr(msg, "role", None)
        if role == "system":
            continue

        if role == "assistant":
            content_items: list[dict[str, Any]] = []
            tool_calls_meta = getattr(msg, "tool_calls_meta", None) or []
            for item in tool_calls_meta:
                name = item.get("name")
                args = item.get("args", {})
                if isinstance(name, str):
                    content_items.extend(_normalize_tool_call(
                        name,
                        args if isinstance(args, dict) else {},
                    ))

            content = getattr(msg, "content", "")
            if isinstance(content, str):
                if not tool_calls_meta:
                    content_items.extend(_parse_tool_calls(content))
                if content.strip():
                    content_items.append({"type": "text", "text": content})
            elif isinstance(content, list):
                content_items.extend(content)

            transcript.append({
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": content_items,
                },
            })
            continue

        content = getattr(msg, "content", "")
        if isinstance(content, list):
            content_items = content
        else:
            content_items = [{"type": "text", "text": str(content)}]

        transcript.append({
            "type": "message",
            "message": {
                "role": "toolResult" if role == "tool" else "user",
                "content": content_items,
            },
        })

    return transcript


def _grade_run(
    task: BenchmarkTaskSpec,
    *,
    messages: list[Any],
    workspace: Path,
    status: str,
) -> tuple[float | None, dict[str, float]]:
    if task.task_type == "builtin":
        return (1.0 if status == "done" else 0.0), {}

    grader = _extract_automated_grader(task.automated_checks)
    if grader is None:
        return None, {}

    transcript = _messages_to_transcript(messages)
    raw_scores = grader(transcript, str(workspace))
    if not isinstance(raw_scores, dict):
        raw_scores = {}
    score, breakdown = _grade_scores(raw_scores)
    return score, breakdown


def _copy_workspace_files(workspace: Path, workspace_files: list[dict[str, Any]], skill_dir: Path) -> None:
    assets_dir = skill_dir / "assets"
    for file_spec in workspace_files:
        if "content" in file_spec and "path" in file_spec:
            dest = workspace / str(file_spec["path"])
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(str(file_spec["content"]), encoding="utf-8")
            continue

        source_name = file_spec.get("source")
        dest_name = file_spec.get("dest")
        if source_name and dest_name:
            source = assets_dir / str(source_name)
            dest = workspace / str(dest_name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)


def _prepare_workspace(task: BenchmarkTaskSpec, skill_dir: Path | None) -> Path:
    if task.task_type == "builtin":
        return ROOT

    if skill_dir is None:
        raise ValueError("skill_dir is required for skill-main tasks")

    workspace_root = ROOT / ".benchmark_workspaces"
    workspace_root.mkdir(parents=True, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix=f"{task.task_id}_", dir=workspace_root)
    workspace = Path(tmpdir)
    _copy_workspace_files(workspace, task.workspace_files or [], skill_dir)
    return workspace


def _load_skill_main_tasks(skill_dir: Path, suite: str) -> list[BenchmarkTaskSpec]:
    tasks_dir = skill_dir / "tasks"
    loaded_tasks = [_parse_skill_task_file(path) for path in sorted(tasks_dir.glob("task_*.md"))]

    if suite == "all":
        selected = loaded_tasks
    elif suite == "automated-only":
        selected = [task for task in loaded_tasks if task.automated_checks]
    else:
        requested = {item.strip() for item in suite.split(",") if item.strip()}
        selected = [
            task for task in loaded_tasks
            if task.task_id in requested or task.name in requested or task.file_path.stem in requested
        ]

    return [
        BenchmarkTaskSpec(
            task_id=task.task_id,
            display_name=task.name or task.task_id,
            prompt=task.prompt,
            task_type="skill-main",
            grading_type=task.grading_type,
            automated_checks=task.automated_checks,
            workspace_files=list(task.workspace_files or []),
        )
        for task in selected
    ]


@dataclass
class _ParsedSkillTask:
    task_id: str
    name: str
    grading_type: str
    prompt: str
    automated_checks: str | None
    workspace_files: list[dict[str, Any]]
    file_path: Path


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    if value == "[]":
        return []
    if value == "{}":
        return {}
    if value in {"null", "Null", "NULL"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_block_scalar(lines: list[str], start_idx: int, parent_indent: int) -> tuple[str, int]:
    i = start_idx
    block_indent: int | None = None
    chunks: list[str] = []
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            chunks.append("")
            i += 1
            continue
        indent = _indent_of(line)
        if indent <= parent_indent:
            break
        if block_indent is None:
            block_indent = indent
        chunks.append(line[block_indent:])
        i += 1
    return "\n".join(chunks), i


def _parse_yaml_subset(lines: list[str], start_idx: int = 0, indent: int = 0) -> tuple[Any, int]:
    i = start_idx
    while i < len(lines) and (not lines[i].strip()):
        i += 1
    if i >= len(lines):
        return {}, i

    if _indent_of(lines[i]) < indent:
        return {}, i

    if lines[i].lstrip().startswith("- "):
        result: list[Any] = []
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                i += 1
                continue
            current_indent = _indent_of(line)
            if current_indent < indent or not line.lstrip().startswith("- "):
                break

            item_text = line.lstrip()[2:]
            if ":" in item_text:
                key, raw = item_text.split(":", 1)
                item: dict[str, Any] = {}
                raw = raw.strip()
                if raw == "|":
                    value, i = _parse_block_scalar(lines, i + 1, current_indent + 1)
                elif raw == "":
                    value, i = _parse_yaml_subset(lines, i + 1, current_indent + 2)
                else:
                    value = _parse_scalar(raw)
                    i += 1
                item[key.strip()] = value

                while i < len(lines):
                    if not lines[i].strip():
                        i += 1
                        continue
                    nested_indent = _indent_of(lines[i])
                    stripped = lines[i].strip()
                    if nested_indent <= current_indent:
                        break
                    if nested_indent == current_indent + 2 and ":" in stripped:
                        nested_key, nested_raw = stripped.split(":", 1)
                        nested_raw = nested_raw.strip()
                        if nested_raw == "|":
                            nested_value, i = _parse_block_scalar(lines, i + 1, nested_indent)
                        elif nested_raw == "":
                            nested_value, i = _parse_yaml_subset(lines, i + 1, nested_indent + 2)
                        else:
                            nested_value = _parse_scalar(nested_raw)
                            i += 1
                        item[nested_key.strip()] = nested_value
                    else:
                        break
                result.append(item)
                continue

            result.append(_parse_scalar(item_text))
            i += 1
        return result, i

    result_dict: dict[str, Any] = {}
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        current_indent = _indent_of(line)
        if current_indent < indent:
            break
        if current_indent > indent:
            i += 1
            continue
        stripped = line.strip()
        if ":" not in stripped:
            i += 1
            continue
        key, raw = stripped.split(":", 1)
        raw = raw.strip()
        if raw == "|":
            value, i = _parse_block_scalar(lines, i + 1, current_indent)
        elif raw == "":
            value, i = _parse_yaml_subset(lines, i + 1, current_indent + 2)
        else:
            value = _parse_scalar(raw)
            i += 1
        result_dict[key.strip()] = value
    return result_dict, i


def _parse_frontmatter(frontmatter_text: str) -> dict[str, Any]:
    parsed, _ = _parse_yaml_subset(frontmatter_text.splitlines(), 0, 0)
    return parsed if isinstance(parsed, dict) else {}


def _parse_markdown_sections(body: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current_section: str | None = None
    current_lines: list[str] = []
    for line in body.splitlines():
        if line.startswith("## "):
            if current_section is not None:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = line[3:].strip()
            current_lines = []
            continue
        if current_section is not None:
            current_lines.append(line)
    if current_section is not None:
        sections[current_section] = "\n".join(current_lines).strip()
    return sections


def _parse_skill_task_file(task_file: Path) -> _ParsedSkillTask:
    content = task_file.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
    if not match:
        raise ValueError(f"No YAML frontmatter found in {task_file}")
    metadata = _parse_frontmatter(match.group(1))
    sections = _parse_markdown_sections(match.group(2))
    return _ParsedSkillTask(
        task_id=str(metadata.get("id", task_file.stem)),
        name=str(metadata.get("name", task_file.stem)),
        grading_type=str(metadata.get("grading_type", "automated")),
        prompt=sections.get("Prompt", "").strip(),
        automated_checks=sections.get("Automated Checks"),
        workspace_files=list(metadata.get("workspace_files", []) or []),
        file_path=task_file,
    )


def _resolve_tasks(
    *,
    task_source: str,
    raw_tasks: list[str] | None,
    skill_dir: Path,
    suite: str,
) -> tuple[list[BenchmarkTaskSpec], Path | None]:
    if task_source == "builtin":
        prompts = list(raw_tasks or BENCHMARK_TASKS)
        return (
            [
                BenchmarkTaskSpec(
                    task_id=f"builtin_{idx:02d}",
                    display_name=prompt,
                    prompt=prompt,
                    task_type="builtin",
                )
                for idx, prompt in enumerate(prompts, start=1)
            ],
            None,
        )

    return _load_skill_main_tasks(skill_dir, suite), skill_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark a fixed single-model workflow against EconomicSupervisorLLM."
    )
    parser.add_argument("--fixed-model", help="Single model for the non-supervisor baseline. Default: active model from config.")
    parser.add_argument("--simple-model", default="gpt-5-mini")
    parser.add_argument("--coding-model", help="Supervisor coding model. Default: fixed model.")
    parser.add_argument("--reasoning-model", help="Supervisor reasoning model. Default: fixed model.")
    parser.add_argument("--task-source", choices=("skill-main", "builtin"), default="skill-main")
    parser.add_argument("--skill-dir", default=str(ROOT / "skill-main"), help="Directory containing the PinchBench skill.")
    parser.add_argument("--suite", default="automated-only", help='For skill-main: "automated-only", "all", or comma-separated task IDs.')
    parser.add_argument("--task", action="append", dest="tasks", help="Benchmark task prompt. Used only with --task-source builtin.")
    parser.add_argument("--price", action="append", dest="prices", help="Override pricing as model=usd_per_1m_tokens.")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument("--no-rethink", action="store_true")
    parser.add_argument("--trajectory", action="store_true")
    parser.add_argument("--output", default="benchmark_economic_supervisor_report.json")
    args = parser.parse_args()

    config = load_config()
    fixed_model = args.fixed_model or get_default_model(config)
    coding_model = args.coding_model or fixed_model
    reasoning_model = args.reasoning_model or fixed_model
    tasks, skill_dir = _resolve_tasks(
        task_source=args.task_source,
        raw_tasks=args.tasks,
        skill_dir=Path(args.skill_dir).resolve(),
        suite=args.suite,
    )
    if not tasks:
        raise SystemExit("No benchmark tasks selected.")
    pricing = _pricing_map(_parse_price_overrides(args.prices))

    fixed_result = asyncio.run(_run_scenario(
        name="fixed-model",
        provider_factory=lambda: _create_fixed_model_provider(fixed_model, pricing),
        tasks=tasks,
        skill_dir=skill_dir,
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
            "task_source": args.task_source,
            "suite": args.suite,
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
        skill_dir=skill_dir,
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
            "task_source": args.task_source,
            "suite": args.suite,
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
