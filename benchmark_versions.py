from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent

INLINE_BENCHMARK = r"""
import asyncio
import json
import os
import sys
from pathlib import Path

workspace = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(workspace / "src"))
os.chdir(workspace)

from clawagents.agent import create_claw_agent
from clawagents.config.config import get_default_model, load_config
from clawagents.providers.llm import create_provider
from clawagents.tools.exec import exec_tools
from clawagents.tools.filesystem import filesystem_tools

async def main():
    tasks = json.loads(os.environ["CLAW_BENCHMARK_TASKS"])
    streaming = os.environ.get("CLAW_BENCHMARK_STREAMING", "1") == "1"
    max_iterations = int(os.environ.get("CLAW_BENCHMARK_MAX_ITERATIONS", "2"))

    config = load_config()
    llm = create_provider(get_default_model(config), config)
    agent = create_claw_agent(
        model=llm,
        tools=filesystem_tools + exec_tools,
        streaming=streaming,
    )

    started = __import__("time").time()
    runs = []
    for task in tasks:
        t0 = __import__("time").time()
        result = await agent.invoke(task, max_iterations=max_iterations)
        runs.append({
            "task": task,
            "elapsed_s": __import__("time").time() - t0,
            "iterations": result.iterations,
            "tool_calls": result.tool_calls,
            "status": result.status,
            "result_preview": (result.result or "")[:200],
        })
    payload = {
        "tasks": runs,
        "total_elapsed_s": __import__("time").time() - started,
        "streaming": streaming,
        "max_iterations": max_iterations,
    }
    print(json.dumps(payload))

asyncio.run(main())
"""


@dataclass
class VersionResult:
    label: str
    workspace: Path
    payload: dict[str, Any]


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _resolve_baseline_ref(explicit_ref: str | None) -> str:
    candidates = [explicit_ref] if explicit_ref else ["origin/main", "main", "HEAD"]
    for candidate in candidates:
        if not candidate:
            continue
        probe = _run(["git", "rev-parse", "--verify", candidate], cwd=ROOT)
        if probe.returncode == 0:
            return candidate
    raise RuntimeError("Could not resolve a baseline ref. Pass --baseline-ref explicitly.")


def _create_worktree(ref: str) -> tuple[Path, Path]:
    temp_root = Path(tempfile.mkdtemp(prefix="clawagents-bench-", dir="/tmp"))
    worktree_dir = temp_root / "baseline"
    add = _run(["git", "worktree", "add", "--detach", str(worktree_dir), ref], cwd=ROOT)
    if add.returncode != 0:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise RuntimeError(add.stderr.strip() or f"git worktree add failed for ref {ref}")
    return temp_root, worktree_dir


def _remove_worktree(temp_root: Path, worktree_dir: Path) -> None:
    try:
        _run(["git", "worktree", "remove", "--force", str(worktree_dir)], cwd=ROOT)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _run_benchmark_for_workspace(
    workspace: Path,
    *,
    label: str,
    tasks: list[str],
    streaming: bool,
    max_iterations: int,
    python_bin: str,
) -> VersionResult:
    env = os.environ.copy()
    env["CLAW_BENCHMARK_TASKS"] = json.dumps(tasks)
    env["CLAW_BENCHMARK_STREAMING"] = "1" if streaming else "0"
    env["CLAW_BENCHMARK_MAX_ITERATIONS"] = str(max_iterations)

    proc = _run(
        [python_bin, "-c", INLINE_BENCHMARK, str(workspace)],
        cwd=ROOT,
        env=env,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"benchmark failed for {label}"
        raise RuntimeError(detail)

    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    payload = json.loads(lines[-1])
    return VersionResult(label=label, workspace=workspace, payload=payload)


def _safe_delta(current: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return ((current - baseline) / baseline) * 100.0


def _build_report(current: VersionResult, baseline: VersionResult, baseline_ref: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    baseline_tasks = {task["task"]: task for task in baseline.payload["tasks"]}
    current_tasks = {task["task"]: task for task in current.payload["tasks"]}

    for task in current_tasks:
        if task not in baseline_tasks:
            continue
        cur = current_tasks[task]
        base = baseline_tasks[task]
        rows.append({
            "task": task,
            "current_elapsed_s": round(cur["elapsed_s"], 3),
            "baseline_elapsed_s": round(base["elapsed_s"], 3),
            "delta_pct": round(_safe_delta(cur["elapsed_s"], base["elapsed_s"]), 2),
            "current_iterations": cur["iterations"],
            "baseline_iterations": base["iterations"],
            "current_tool_calls": cur["tool_calls"],
            "baseline_tool_calls": base["tool_calls"],
            "current_status": cur["status"],
            "baseline_status": base["status"],
        })

    return {
        "baseline_ref": baseline_ref,
        "current": {
            "label": current.label,
            "workspace": str(current.workspace),
            "total_elapsed_s": round(current.payload["total_elapsed_s"], 3),
        },
        "baseline": {
            "label": baseline.label,
            "workspace": str(baseline.workspace),
            "total_elapsed_s": round(baseline.payload["total_elapsed_s"], 3),
        },
        "tasks": rows,
    }


def _print_report(report: dict[str, Any]) -> None:
    print(f"Baseline ref: {report['baseline_ref']}")
    print(
        f"Current total: {report['current']['total_elapsed_s']:.3f}s | "
        f"Baseline total: {report['baseline']['total_elapsed_s']:.3f}s"
    )
    print("")
    print(
        "| Task | Current (s) | Baseline (s) | Delta % | "
        "Current Iter | Baseline Iter | Current Tools | Baseline Tools |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in report["tasks"]:
        print(
            f"| {row['task']} | {row['current_elapsed_s']:.3f} | {row['baseline_elapsed_s']:.3f} | "
            f"{row['delta_pct']:+.2f}% | {row['current_iterations']} | {row['baseline_iterations']} | "
            f"{row['current_tool_calls']} | {row['baseline_tool_calls']} |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark current workspace against a baseline git ref.")
    parser.add_argument("--baseline-ref", help="Git ref to compare against. Default: origin/main, then main, then HEAD.")
    parser.add_argument("--output", default="benchmark_compare_report.json", help="Path to write JSON report.")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to use for both runs.")
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--no-streaming", action="store_true", help="Disable streaming during the benchmark runs.")
    parser.add_argument("--task", action="append", dest="tasks", help="Override benchmark tasks. Can be passed multiple times.")
    args = parser.parse_args()

    tasks = args.tasks or [
        "Read pyproject.toml and tell me the version number.",
        "Use the execute tool to run `echo benchmark` and return the output.",
    ]
    baseline_ref = _resolve_baseline_ref(args.baseline_ref)

    temp_root, baseline_dir = _create_worktree(baseline_ref)
    try:
        current_result = _run_benchmark_for_workspace(
            ROOT,
            label="current",
            tasks=tasks,
            streaming=not args.no_streaming,
            max_iterations=args.max_iterations,
            python_bin=args.python,
        )
        baseline_result = _run_benchmark_for_workspace(
            baseline_dir,
            label="baseline",
            tasks=tasks,
            streaming=not args.no_streaming,
            max_iterations=args.max_iterations,
            python_bin=args.python,
        )
        report = _build_report(current_result, baseline_result, baseline_ref)
        output_path = Path(args.output)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        _print_report(report)
        print(f"\nSaved JSON report to {output_path.resolve()}")
    finally:
        _remove_worktree(temp_root, baseline_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
