"""Compose an agent without editing global state.

A model adapter, tool list, profile loader, checkpointer, and graph factory are
ordinary constructor arguments. LangGraph owns the execution/checkpoint protocol;
this class supplies run lifecycle and dependency wiring.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, cast

from langgraph.checkpoint.base import BaseCheckpointSaver

from market_analyst.llm import ModelSettings
from market_analyst.runtime.harness import harness_run
from market_analyst.schemas import ExecutionMode, UserProfile
from market_analyst.tools.registry import RESEARCH_TOOLS
from market_analyst.workflows.analysis_workflow import create_graph, run_analysis


@dataclass
class AgentHarness:
    model: ModelSettings = field(default_factory=ModelSettings.from_env)
    tools: tuple = RESEARCH_TOOLS
    checkpointer: BaseCheckpointSaver | None = None
    profile_loader: Callable[[str], UserProfile] = lambda _: UserProfile()
    graph_factory: Callable[..., Any] = create_graph

    def run(self, query: str, *, thread_id: str, user_id: str = "default", mode: ExecutionMode | None = None, retry: bool = False) -> dict:
        with harness_run(thread_id) as context:
            result = run_analysis(
                query,
                user_id,
                thread_id,
                self.checkpointer,
                mode,
                model_settings=self.model,
                tools=list(self.tools),
                profile_loader=self.profile_loader,
                retry_delivery=retry,
                graph_factory=self.graph_factory,
            )
            context.final_state = result["state"]
            return cast(dict, result)
