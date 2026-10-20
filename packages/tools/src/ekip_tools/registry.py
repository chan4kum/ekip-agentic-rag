"""The tool registry: a reviewed, version-controlled allowlist of what agents may call.

`tools/registry.yaml` is the source of truth. A tool that is not listed (or is disabled) does not
exist as far as the gateway is concerned; the model cannot add, change or discover tools at run
time. Specs are validated when loaded so a bad entry stops startup instead of failing a request.
"""

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ekip_contracts.tools import ToolSpec

from ekip_tools.schema import check_schema


class RegistryError(ValueError):
    pass


class ToolRegistry:
    def __init__(self, specs: Iterable[ToolSpec]) -> None:
        self._specs: dict[str, ToolSpec] = {}
        for spec in specs:
            if spec.name in self._specs:
                raise RegistryError(f"duplicate tool: {spec.name}")
            try:
                check_schema(spec.input_schema)
            except ValueError as exc:
                raise RegistryError(f"{spec.name}: {exc}") from exc
            self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        """The enabled spec called `name`, else None (disabled tools are invisible)."""
        spec = self._specs.get(name)
        return spec if spec is not None and spec.enabled else None

    def enabled(self) -> list[ToolSpec]:
        return [s for s in self._specs.values() if s.enabled]

    def backends_in_use(self) -> set[str]:
        return {s.backend for s in self.enabled()}


def registry_from_mapping(data: Mapping[str, Any]) -> ToolRegistry:
    tools = data.get("tools")
    if not isinstance(tools, list) or not tools:
        raise RegistryError("registry needs a non-empty 'tools' list")
    try:
        return ToolRegistry(ToolSpec.model_validate(entry) for entry in tools)
    except ValueError as exc:  # pydantic ValidationError is a ValueError
        raise RegistryError(str(exc)) from exc


def load_registry_file(path: Path) -> ToolRegistry:
    """Read `tools/registry.yaml`. PyYAML is imported here, not at module import, so the rest of
    the package works without it (it is a dev dependency of the workspace today)."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RegistryError("PyYAML is required to load the registry file") from exc
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise RegistryError("registry file must be a mapping")
    return registry_from_mapping(loaded)
