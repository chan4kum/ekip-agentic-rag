"""Regenerate the prompt section of ops/model-registry.yaml from the prompts in the code.

    uv run --no-sync python scripts/update_registry.py          # rewrite the file
    uv run --no-sync python scripts/update_registry.py --check  # exit 1 if it is out of date

Run it after bumping a prompt version (and after the evaluations for that change passed). The
`models` section is hand-maintained and is preserved as-is.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml
from ekip_agents.groundedness import GROUND_PROMPT_VERSION, GROUND_SYSTEM_PROMPT
from ekip_agents.planner import PLAN_PROMPT_VERSION, plan_system_prompt
from ekip_agents.prompts import PROMPT_VERSION, SYSTEM_PROMPT
from ekip_agents.rewrite import REWRITE_PROMPT_VERSION, REWRITE_SYSTEM_PROMPT
from ekip_observability.registry import check_prompts, prompt_entry

REGISTRY_PATH = Path("ops/model-registry.yaml")
PLAN_SUBTASKS = 4  # the planner's default; the registry hashes the prompt rendered with it
HEADER = (
    "# Model and prompt registry (data). The `prompts` section is generated:\n"
    "#   uv run --no-sync python scripts/update_registry.py\n"
    "# CI fails when a prompt's text changes without a version bump\n"
    "# (tests/unit/test_llmops_registry.py).\n"
)


def collect_prompts() -> dict[str, tuple[str, str]]:
    """prompt name -> (version, rendered text)."""
    return {
        "answer": (PROMPT_VERSION, SYSTEM_PROMPT),
        "rewrite": (REWRITE_PROMPT_VERSION, REWRITE_SYSTEM_PROMPT),
        "plan": (PLAN_PROMPT_VERSION, plan_system_prompt(PLAN_SUBTASKS)),
        "groundedness": (GROUND_PROMPT_VERSION, GROUND_SYSTEM_PROMPT),
    }


def render(existing: dict[str, Any]) -> str:
    document = {
        "version": 1,
        "models": existing.get("models", {}),
        "prompts": {n: prompt_entry(v, t) for n, (v, t) in sorted(collect_prompts().items())},
    }
    return HEADER + yaml.safe_dump(document, sort_keys=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--path", type=Path, default=REGISTRY_PATH)
    args = parser.parse_args()
    existing: dict[str, Any] = yaml.safe_load(args.path.read_text()) if args.path.exists() else {}
    if args.check:
        problems = check_prompts(existing, collect_prompts())
        for p in problems:
            print(f"registry: {p}")
        return 1 if problems else 0
    args.path.write_text(render(existing))
    print(f"wrote {args.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
