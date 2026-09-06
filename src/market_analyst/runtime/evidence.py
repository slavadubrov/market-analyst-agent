"""Evidence is tool output, not the model's description of its own research."""

from datetime import datetime, timezone
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


def record_evidence(tool: str, result: Any) -> dict:
    return {"tool": tool, "observed_at": datetime.now(timezone.utc).isoformat(), "result": result}


def validate_evidence(evidence: list[dict], *, max_age_seconds: int = 86400) -> list[str]:
    """Validate collection freshness; source publication dates remain separate."""
    if not evidence:
        return ["No source evidence was collected"]
    errors = []
    now = datetime.now(timezone.utc)
    for item in evidence:
        try:
            age = (now - datetime.fromisoformat(item["observed_at"])).total_seconds()
            if age < -60 or age > max_age_seconds:
                errors.append("Evidence collection timestamp is stale or in the future")
            if not item["result"] or str(item["result"]).startswith(("Error", "Blocked")):
                errors.append("Evidence contains an unsuccessful tool result")
        except (KeyError, ValueError, TypeError):
            errors.append("Evidence has no valid collection timestamp")
    return list(dict.fromkeys(errors))


class ToolEvidenceCollector(BaseCallbackHandler):
    """Capture actual tool results before context compaction can remove messages."""

    def __init__(self):
        self.records = []

    def on_tool_end(self, output, *, name=None, **kwargs):
        value = getattr(output, "content", output)
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        self.records.append(record_evidence(name or getattr(output, "name", None) or "tool", value))
