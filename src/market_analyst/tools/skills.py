"""Skills tool — provides domain expertise via SKILL.md files.

Skills use progressive disclosure: only metadata (~100 tokens) loads at startup,
full instructions load only when the agent activates a skill. This is fundamentally
different from data tools — skills provide *methodology*, not data.
"""

from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Default skills directory (relative to project root)
_SKILLS_DIR = Path(__file__).parent.parent.parent.parent / "skills"


class SkillMetadata(BaseModel):
    """Parsed SKILL.md frontmatter."""

    name: str
    description: str
    path: Path


def _parse_frontmatter(text: str) -> dict[str, str]:
    """Parse YAML frontmatter from a SKILL.md file.

    Handles simple key: value pairs between --- delimiters.
    No PyYAML dependency needed.
    """
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    frontmatter = {}
    for line in parts[1].strip().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            frontmatter[key.strip()] = value.strip()
    return frontmatter


def _get_body(text: str) -> str:
    """Extract the markdown body after frontmatter."""
    parts = text.split("---", 2)
    if len(parts) < 3:
        return text
    return parts[2].strip()


def load_skill_metadata(skills_dir: Path | None = None) -> list[SkillMetadata]:
    """Load metadata from all SKILL.md files in the skills directory.

    Only reads frontmatter (~100 tokens per skill) for progressive disclosure.
    """
    directory = skills_dir or _SKILLS_DIR
    if not directory.exists():
        return []
    skills = []
    for path in sorted(directory.glob("*.md")):
        text = path.read_text()
        meta = _parse_frontmatter(text)
        if "name" in meta and "description" in meta:
            skills.append(
                SkillMetadata(
                    name=meta["name"],
                    description=meta["description"],
                    path=path,
                )
            )
    return skills


def get_skill_descriptions(skills_dir: Path | None = None) -> str:
    """Get a compact summary of available skills for the system prompt.

    Returns a string like:
      Available skills (use `use_skill` to activate):
      - earnings_analysis: Step-by-step methodology for ...
      - sector_comparison: Framework for comparing ...
    """
    skills = load_skill_metadata(skills_dir)
    if not skills:
        return ""
    lines = ["Available skills (use `use_skill` tool to activate):"]
    for s in skills:
        lines.append(f"- {s.name}: {s.description}")
    return "\n".join(lines)


class SkillQuery(BaseModel):
    """Input schema for the use_skill tool."""

    skill_name: str = Field(
        description=("Name of the skill to activate. Available skills: earnings_analysis, sector_comparison"),
    )


@tool(args_schema=SkillQuery)
def use_skill(skill_name: str) -> str:
    """Activate a skill to get expert methodology and step-by-step instructions.

    Use this tool when you need a structured approach or playbook for a
    specific type of analysis. Skills provide expertise (how to analyze),
    not data. After activating a skill, follow its instructions using
    your data-fetching tools.

    Available skills:
    - earnings_analysis: Methodology for analyzing quarterly earnings
    - sector_comparison: Framework for comparing stocks against peers
    """
    skills = load_skill_metadata()
    skill_map = {s.name: s for s in skills}

    if skill_name not in skill_map:
        available = ", ".join(skill_map.keys())
        return f"Unknown skill '{skill_name}'. Available skills: {available}"

    text = skill_map[skill_name].path.read_text()
    return _get_body(text)
