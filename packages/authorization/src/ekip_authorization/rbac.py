"""Role-based access control with a mandatory tenant check. In-memory, static, fail closed."""

from collections.abc import Mapping

from ekip_contracts.authorization import Action, Decision, Resource
from ekip_contracts.identity import SecurityContext
from ekip_observability.logging import get_logger

log = get_logger("ekip.authorization")

# role -> actions it grants. Roles come from the validated token; anything not listed grants
# nothing. Keep this small and reviewed: it is the whole permission model for now.
DEFAULT_POLICY: Mapping[str, frozenset[Action]] = {
    "ekip-user": frozenset({Action.DOCUMENTS_SEARCH, Action.DOCUMENTS_READ}),
    "ekip-admin": frozenset({Action.DOCUMENTS_SEARCH, Action.DOCUMENTS_READ, Action.ADMIN_MANAGE}),
}


class RbacAuthorizer:
    def __init__(self, policy: Mapping[str, frozenset[Action]] = DEFAULT_POLICY) -> None:
        self._policy = dict(
            policy
        )  # copy: later mutation of the caller's dict cannot change policy

    async def authorize(
        self, context: SecurityContext, action: Action, resource: Resource
    ) -> Decision:
        try:
            return self._decide(context, action, resource)
        except Exception as exc:  # a bug in policy code must never turn into an allow
            log.error("authorization_error", error_type=type(exc).__name__)
            return Decision(allowed=False, reason="policy_error")

    def _decide(self, context: SecurityContext, action: Action, resource: Resource) -> Decision:
        try:
            action = Action(action)  # tolerate a raw string; reject anything not in the enum
        except ValueError:
            return Decision(allowed=False, reason="unknown_action")
        # Tenant isolation is checked FIRST and for every role, admins included (I-4).
        if resource.tenant_id != context.tenant_id:
            return Decision(allowed=False, reason="tenant_mismatch")
        granted = any(action in self._policy.get(role, frozenset()) for role in context.roles)
        if not granted:
            return Decision(allowed=False, reason="no_role_grants_action")
        return Decision(allowed=True, reason="allowed")
