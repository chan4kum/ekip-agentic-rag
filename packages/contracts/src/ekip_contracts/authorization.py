"""Port for authorization decisions. Implementations live in ekip_authorization (or an OpenFGA/OPA
adapter later). The LLM never makes or influences these decisions (I-2)."""

from enum import StrEnum
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from ekip_contracts.identity import SecurityContext


class Action(StrEnum):
    """Every operation that needs a permission. Adding one is a deliberate, reviewed change."""

    DOCUMENTS_SEARCH = "documents:search"
    DOCUMENTS_READ = "documents:read"
    ADMIN_MANAGE = "admin:manage"


class Resource(BaseModel):
    """What the caller wants to act on. `tenant_id` is the OWNER of the resource, taken from
    our data (never from the request), so a mismatch with the caller's tenant is detectable."""

    model_config = ConfigDict(frozen=True)

    kind: str
    tenant_id: str
    id: str | None = None


DecisionReason = Literal[
    "allowed",
    "no_role_grants_action",
    "tenant_mismatch",
    "unknown_action",
    "policy_error",
]


class Decision(BaseModel):
    model_config = ConfigDict(frozen=True)

    allowed: bool
    reason: DecisionReason


class Authorizer(Protocol):
    async def authorize(
        self, context: SecurityContext, action: Action, resource: Resource
    ) -> Decision:
        """Return a decision. Anything uncertain must be a deny (fail closed, I-1)."""
        ...
