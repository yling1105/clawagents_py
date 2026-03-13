"""Example: Dynamic budget-aware model routing."""
import asyncio

from clawagents import BudgetLedger, EconomicSupervisorLLM, create_claw_agent
from clawagents.config.config import load_config


async def main():
    config = load_config()
    ledger = BudgetLedger(
        total_budget_usd=2.50,
        pricing_usd_per_1m_tokens={
            "gpt-5-mini": 0.50,
            "gpt-5.3-codex": 5.00,
            "gpt-5.4": 12.00,
        },
    )
    supervisor = EconomicSupervisorLLM(
        config,
        ledger,
        simple_model="gpt-5-mini",
        coding_model="gpt-5.3-codex",
        reasoning_model="gpt-5.4",
    )

    agent = create_claw_agent(
        model=supervisor,
        rethink=True,
        trajectory=True,
    )
    result = await agent.invoke(
        "Inspect the repository, identify the failing test, fix the bug, and summarize the changes."
    )

    print(result.result)
    print("\nFinancial report:")
    print(result.financial_report or supervisor.get_financial_report())


if __name__ == "__main__":
    asyncio.run(main())
