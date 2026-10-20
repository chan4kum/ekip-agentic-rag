"""Prompt text may only change together with its version (ops/model-registry.yaml)."""

from pathlib import Path

import yaml
from ekip_observability.registry import check_prompts, prompt_entry, sha256_text
from script_loader import ROOT, load_script


def registry() -> dict[str, object]:
    data = yaml.safe_load((ROOT / "ops" / "model-registry.yaml").read_text())
    assert isinstance(data, dict)
    return data


def test_checked_in_registry_matches_the_prompts_in_the_code() -> None:
    """If this fails: re-run the evaluations, bump the prompt version if the text changed, then
    `uv run --no-sync python scripts/update_registry.py` and commit the result."""
    current = load_script("update_registry").collect_prompts()
    assert check_prompts(registry(), current) == []


def test_registry_declares_models() -> None:
    models = registry()["models"]
    assert isinstance(models, dict)
    assert models["chat"]["name"] and models["embedding"]["name"]
    assert models["embedding"]["dimensions"] > 0


def test_every_prompt_has_a_version_and_a_sha256() -> None:
    prompts = registry()["prompts"]
    assert isinstance(prompts, dict) and prompts
    for entry in prompts.values():
        assert entry["version"]
        assert len(entry["sha256"]) == 64


def test_editing_text_without_a_version_bump_is_reported() -> None:
    reg = {"prompts": {"answer": prompt_entry("answer-v3", "old text")}}
    problems = check_prompts(reg, {"answer": ("answer-v3", "new text")})
    assert len(problems) == 1 and "bump the version" in problems[0]


def test_version_bump_with_stale_registry_asks_for_regeneration() -> None:
    reg = {"prompts": {"answer": prompt_entry("answer-v3", "text")}}
    problems = check_prompts(reg, {"answer": ("answer-v4", "text")})
    assert len(problems) == 1 and "update_registry" in problems[0]


def test_unchanged_prompt_passes_and_unknown_ones_are_reported() -> None:
    reg = {"prompts": {"a": prompt_entry("v1", "x"), "gone": prompt_entry("v1", "y")}}
    assert check_prompts(reg, {"a": ("v1", "x")}) == [
        "gone: in the registry but no longer in the code"
    ]
    assert "not in the registry" in check_prompts(reg, {"a": ("v1", "x"), "b": ("v1", "z")})[0]


def test_malformed_registry_is_a_problem_not_a_crash() -> None:
    assert check_prompts({"prompts": []}, {}) == ["registry: 'prompts' must be a mapping"]


def test_hash_is_sha256_of_utf8() -> None:
    assert sha256_text("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert sha256_text("é") != sha256_text("e")


def test_update_script_roundtrip_is_stable(tmp_path: Path) -> None:
    mod = load_script("update_registry")
    path = tmp_path / "reg.yaml"
    path.write_text(
        yaml.safe_dump({"version": 1, "models": {"chat": {"name": "m"}}, "prompts": {}})
    )
    existing = yaml.safe_load(path.read_text())
    path.write_text(mod.render(existing))
    again = yaml.safe_load(path.read_text())
    assert again["models"] == {"chat": {"name": "m"}}
    assert mod.render(again) == path.read_text()
    assert check_prompts(again, mod.collect_prompts()) == []
