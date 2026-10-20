"""Prompt registry checks: a prompt's text may only change together with its version.

`ops/model-registry.yaml` stores, per prompt, the version and the sha256 of its text. CI compares
that with the prompts compiled into the code:
- same version, different text  -> someone edited a prompt without bumping the version (error);
- different version             -> the registry is stale: run scripts/update_registry.py and commit
                                   (this is the moment evaluations must have been re-run).
Pure functions over plain mappings; reading YAML / importing prompts is the caller's job.
"""

import hashlib
from collections.abc import Mapping
from typing import Any


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_entry(version: str, text: str) -> dict[str, str]:
    return {"version": version, "sha256": sha256_text(text)}


def check_prompts(registry: Mapping[str, Any], current: Mapping[str, tuple[str, str]]) -> list[str]:
    """`current` maps prompt name -> (version, text). Returns human-readable problems."""
    problems: list[str] = []
    registered = registry.get("prompts", {})
    if not isinstance(registered, Mapping):
        return ["registry: 'prompts' must be a mapping"]
    for name, (version, text) in sorted(current.items()):
        entry = registered.get(name)
        if not isinstance(entry, Mapping):
            problems.append(f"{name}: not in the registry (run scripts/update_registry.py)")
            continue
        if entry.get("version") != version:
            problems.append(
                f"{name}: code is at {version}, registry at {entry.get('version')} "
                "(re-run evaluations, then scripts/update_registry.py)"
            )
        elif entry.get("sha256") != sha256_text(text):
            problems.append(
                f"{name}: text changed but version is still {version} - bump the version"
            )
    for name in sorted(set(registered) - set(current)):
        problems.append(f"{name}: in the registry but no longer in the code")
    return problems
