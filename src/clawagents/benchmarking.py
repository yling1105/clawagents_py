from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

from clawagents.agent import create_claw_agent
from clawagents.config.config import get_default_model, load_config
from clawagents.providers.llm import create_provider
from clawagents.tools.exec import exec_tools
from clawagents.tools.filesystem import filesystem_tools

BENCHMARK_TASKS_SIMPLE: list[str] = [
    "Read pyproject.toml and tell me the version number.",
    "Read README.md and summarize the project in one sentence.",
    "Use the execute tool to run `pwd` and return the repository root directory.",
]

BENCHMARK_TASKS_CODING: list[str] = [
    "Inspect tests/test_economic_supervisor.py and summarize what behavior it validates.",
    "Use the execute tool to run `pytest tests/test_economic_supervisor.py -q` and report whether the pytest run passed.",
    "Read src/clawagents/providers/economic.py and summarize how `_assess_complexity_and_route` chooses between models.",
]

BENCHMARK_TASKS_REASONING: list[str] = [
    "Compare approaches: explain the tradeoff between using one fixed model and EconomicSupervisorLLM for this repository's benchmark.",
    "Read benchmark_economic_supervisor.py and propose an implementation plan to make the benchmark statistically stronger.",
    "Why does this benchmark need both latency and cost metrics? Give a design-level explanation based on the current benchmark code.",
]

BENCHMARK_TASKS: list[str] = [
    *BENCHMARK_TASKS_SIMPLE,
    *BENCHMARK_TASKS_CODING,
    *BENCHMARK_TASKS_REASONING,
]


@dataclass
class BenchmarkRun:
    task: str
    elapsed_s: float
    iterations: int
    tool_calls: int
    status: str
    result_preview: str


def create_benchmark_agent(*, streaming: bool = True):
    config = load_config()
    llm = create_provider(get_default_model(config), config)
    tools = filesystem_tools + exec_tools
    return create_claw_agent(model=llm, tools=tools, streaming=streaming)


async def run_benchmark_tasks(
    tasks: list[str] | None = None,
    *,
    streaming: bool = True,
    max_iterations: int = 2,
) -> dict[str, Any]:
    task_list = list(tasks or BENCHMARK_TASKS)
    agent = create_benchmark_agent(streaming=streaming)

    runs: list[BenchmarkRun] = []
    started = time.time()
    for task in task_list:
        t0 = time.time()
        result = await agent.invoke(task, max_iterations=max_iterations)
        runs.append(BenchmarkRun(
            task=task,
            elapsed_s=time.time() - t0,
            iterations=result.iterations,
            tool_calls=result.tool_calls,
            status=result.status,
            result_preview=(result.result or "")[:200],
        ))

    return {
        "tasks": [asdict(run) for run in runs],
        "total_elapsed_s": time.time() - started,
        "streaming": streaming,
        "max_iterations": max_iterations,
    }
