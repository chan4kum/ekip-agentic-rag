"""Tool authorization. The model never decides this (I-2)."""

from collections.abc import Mapping
from typing import Any

from ekip_contracts.identity import SecurityContext
from ekip_contracts.tools import ToolDecision, ToolSpec


class RoleToolAuthorizer:
    """Allow when the caller holds ANY of the tool's required roles. No roles, no access."""

    async def authorize(
        self, context: SecurityContext, spec: ToolSpec, arguments: Mapping[str, Any]
    ) -> ToolDecision:
        if context.roles.intersection(spec.required_roles):
            return ToolDecision(allowed=True)
        return ToolDecision(allowed=False, reason="missing_role")
