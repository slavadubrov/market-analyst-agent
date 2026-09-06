"""Opt-in paid model smoke test with a real quote and no search-key dependency."""

import argparse
import uuid

from dotenv import load_dotenv

from market_analyst.harness import AgentHarness
from market_analyst.llm import ModelSettings
from market_analyst.schemas import ExecutionMode
from market_analyst.tools.stock import get_stock_snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--mode", choices=["flash", "deep"], default="flash")
    args = parser.parse_args()
    load_dotenv()
    harness = AgentHarness(model=ModelSettings.from_env(args.model), tools=(get_stock_snapshot,))
    result = harness.run(
        "Fetch the NVDA stock snapshot and write a short report. Use only get_stock_snapshot. "
        "For deep mode use exactly one research step. State missing news, market timestamp, and other evidence as limitations.",
        thread_id="live-smoke-" + uuid.uuid4().hex,
        mode=ExecutionMode.FLASH_BRIEFING if args.mode == "flash" else ExecutionMode.DEEP_RESEARCH,
    )
    assert result["draft_report"] is not None
    assert result["state"]["evidence"]
    print("SMOKE PASS:", args.mode, "draft and tool evidence present; evaluator:", result["state"]["evaluator_verdict"])


if __name__ == "__main__":
    main()
