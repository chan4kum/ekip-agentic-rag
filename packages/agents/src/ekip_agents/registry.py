"""The static agent registry: which agents exist, what each may do, and its limits.

It is a version-controlled YAML file (`agents/registry.yaml`) loaded and strictly validated at
startup. A bad file fails the app start, not the first request. The registry is DATA about
agents (domain, capabilities, permissions they require, tools they may use, budgets); nothing a
model says can add to it. Dynamic discovery is deliberately not built (ADR-008).
"""

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ekip_agents.budget import Budget

AGENT_ID = re.compile(r"^[a-z][a-z0-9_-]{1,39}$")
DOMAIN = re.compile(r"^[a-z][a-z0-9_-]{1,39}$")
NAME = re.compile(r"^[a-z][a-z0-9_.:-]{1,63}$")  # capability / permission / tool names
SEMVER = re.compile(r"^\d{1,4}\.\d{1,4}\.\d{1,4}$")
MAX_AGENTS = 100
MAX_LIST = 32

_TOP_KEYS = frozenset({"version", "agents"})
_AGENT_KEYS = frozenset(
    {
        "id",
        "version",
        "domain",
        "description",
        "capabilities",
        "required_permissions",
        "approved_tools",
        "requires_approval",
        "budget",
    }
)
_BUDGET_KEYS = frozenset(
    {
        "max_steps",
        "max_retrievals",
        "max_tool_calls",
        "max_model_calls",
        "max_tokens",
        "max_seconds",
    }
)


class RegistryError(Exception):
    """The registry file is missing, malformed, or breaks a rule. Messages name the agent id and
    the rule, never file contents."""


@dataclass(frozen=True)
class AgentSpec:
    id: str
    version: str
    domain: str
    description: str
    capabilities: tuple[str, ...]
    required_permissions: tuple[str, ...] = ()
    approved_tools: tuple[str, ...] = ()
    requires_approval: bool = False
    budget: Budget = field(default_factory=Budget)


@dataclass(frozen=True)
class AgentRegistry:
    agents: tuple[AgentSpec, ...]

    def __iter__(self) -> Iterator[AgentSpec]:
        return iter(self.agents)

    def __len__(self) -> int:
        return len(self.agents)

    def get(self, agent_id: str) -> AgentSpec | None:
        return next((a for a in self.agents if a.id == agent_id), None)

    def by_domain(self, domain: str) -> tuple[AgentSpec, ...]:
        return tuple(a for a in self.agents if a.domain == domain)

    def with_capability(self, capability: str) -> tuple[AgentSpec, ...]:
        return tuple(a for a in self.agents if capability in a.capabilities)

    def allows_tool(self, agent_id: str, tool: str) -> bool:
        """Fail closed: unknown agent or tool means no."""
        spec = self.get(agent_id)
        return spec is not None and tool in spec.approved_tools


def _names(agent_id: str, key: str, value: Any, *, required: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > MAX_LIST or (required and not value):
        raise RegistryError(f"{agent_id}: {key} must be a list of 1..{MAX_LIST} names")
    if not all(isinstance(v, str) and NAME.fullmatch(v) for v in value):
        raise RegistryError(f"{agent_id}: {key} has an invalid name")
    if len(set(value)) != len(value):
        raise RegistryError(f"{agent_id}: {key} has duplicates")
    return tuple(value)


def _budget(agent_id: str, raw: Any) -> Budget:
    if raw is None:
        return Budget()
    if not isinstance(raw, Mapping) or set(raw) - _BUDGET_KEYS:
        raise RegistryError(f"{agent_id}: unexpected budget fields")
    for key, value in raw.items():
        ok = isinstance(value, int | float) and not isinstance(value, bool)
        if not ok or (key != "max_seconds" and not isinstance(value, int)):
            raise RegistryError(f"{agent_id}: budget {key} must be a number")
    try:
        return Budget(**raw)
    except ValueError as exc:
        raise RegistryError(f"{agent_id}: {exc}") from exc


def _agent(raw: Any) -> AgentSpec:
    if not isinstance(raw, Mapping):
        raise RegistryError("each agent must be a mapping")
    agent_id = raw.get("id")
    if not isinstance(agent_id, str) or not AGENT_ID.fullmatch(agent_id):
        raise RegistryError("agent id is missing or invalid")
    if set(raw) - _AGENT_KEYS:
        raise RegistryError(f"{agent_id}: unexpected fields")
    version, domain, description = raw.get("version"), raw.get("domain"), raw.get("description")
    if not isinstance(version, str) or not SEMVER.fullmatch(version):
        raise RegistryError(f"{agent_id}: version must look like 1.0.0")
    if not isinstance(domain, str) or not DOMAIN.fullmatch(domain):
        raise RegistryError(f"{agent_id}: domain is missing or invalid")
    if not isinstance(description, str) or not 1 <= len(description) <= 200:
        raise RegistryError(f"{agent_id}: description must be 1..200 characters")
    requires_approval = raw.get("requires_approval", False)
    if not isinstance(requires_approval, bool):
        raise RegistryError(f"{agent_id}: requires_approval must be true or false")
    return AgentSpec(
        id=agent_id,
        version=version,
        domain=domain,
        description=description,
        capabilities=_names(agent_id, "capabilities", raw.get("capabilities"), required=True),
        required_permissions=_names(
            agent_id, "required_permissions", raw.get("required_permissions", [])
        ),
        approved_tools=_names(agent_id, "approved_tools", raw.get("approved_tools", [])),
        requires_approval=requires_approval,
        budget=_budget(agent_id, raw.get("budget")),
    )


def parse_registry(data: Any) -> AgentRegistry:
    if not isinstance(data, Mapping) or set(data) - _TOP_KEYS:
        raise RegistryError("registry must be a mapping with 'version' and 'agents'")
    if data.get("version") != 1:
        raise RegistryError("unsupported registry version")
    raw = data.get("agents")
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_AGENTS:
        raise RegistryError(f"agents must be a list of 1..{MAX_AGENTS} entries")
    agents = tuple(_agent(item) for item in raw)
    ids = [a.id for a in agents]
    if len(set(ids)) != len(ids):
        raise RegistryError("duplicate agent id")
    return AgentRegistry(agents)


def load_registry(path: Path | str) -> AgentRegistry:
    """Read and validate a registry file. Raises RegistryError for any problem."""
    try:
        text = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(text)  # safe_load: no arbitrary Python objects from a data file
    except (OSError, UnicodeDecodeError) as exc:
        raise RegistryError(f"cannot read registry: {type(exc).__name__}") from exc
    except yaml.YAMLError as exc:
        raise RegistryError("registry is not valid YAML") from exc
    return parse_registry(data)
