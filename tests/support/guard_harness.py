"""Shared fixtures for the Phase 9 attack tests: a tenant/ACL-honouring fake index, a scripted
model that records what it was shown, and a helper that builds the REAL agent graph with the
guardrail decorators in place."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ekip_agents import Graph, RuleBasedRouter, build_agent_graph
from ekip_contracts.conversation import ConversationStore
from ekip_contracts.identity import SecurityContext
from ekip_contracts.llm import ChatMessage
from ekip_contracts.retrieval import RetrievedChunk, SearchScope
from ekip_guardrails import (
    GuardedChatModel,
    GuardedRetriever,
    GuardrailPolicy,
    InMemoryAuditLogger,
    PolicyGuardrailEngine,
    prompt_shingles,
)

ALICE = SecurityContext(
    subject="u-alice", tenant_id="tenant-a", principal="alice", groups=frozenset({"eng"})
)
BOB_HR = SecurityContext(
    subject="u-bob", tenant_id="tenant-a", principal="bob", groups=frozenset({"hr"})
)


@dataclass(frozen=True)
class IndexedDoc:
    chunk: RetrievedChunk
    tenant_id: str
    users: frozenset[str] = frozenset()
    groups: frozenset[str] = frozenset()


def doc(
    text: str,
    *,
    chunk_id: str = "c1",
    tenant: str = "tenant-a",
    users: frozenset[str] = frozenset({"alice"}),
    groups: frozenset[str] = frozenset(),
    source_uri: str = "",
) -> IndexedDoc:
    chunk = RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        text=text,
        score=1.0,
        title="Manual",
        source_uri=source_uri,
    )
    return IndexedDoc(chunk, tenant, users, groups)


class AclIndex:
    """Honours SearchScope exactly like a real search adapter: tenant AND ACL inside the 'query'.
    An empty ACL matches nobody (fail closed)."""

    def __init__(self, docs: Sequence[IndexedDoc]) -> None:
        self.docs = list(docs)
        self.scopes: list[SearchScope] = []
        self.queries: list[str] = []

    async def search(
        self, query: str, *, scope: SearchScope, limit: int = 5
    ) -> list[RetrievedChunk]:
        self.scopes.append(scope)
        self.queries.append(query)
        return [
            d.chunk
            for d in self.docs
            if d.tenant_id == scope.tenant_id
            and (scope.user in d.users or bool(scope.groups & d.groups))
        ][:limit]


class ScriptedModel:
    """Records every prompt; the reply is a fixed string or computed from the prompt."""

    def __init__(
        self, reply: str | Callable[[Sequence[ChatMessage]], str] = "Torque is 140 Nm [1]."
    ):
        self.reply = reply
        self.calls: list[list[ChatMessage]] = []

    @property
    def model_name(self) -> str:
        return "scripted"

    async def complete(
        self, messages: Sequence[ChatMessage], *, max_tokens: int = 800, temperature: float = 0.0
    ) -> str:
        self.calls.append(list(messages))
        return self.reply(messages) if callable(self.reply) else self.reply

    def prompt_text(self) -> str:
        return "\n".join(m.content for call in self.calls for m in call)


@dataclass
class Rig:
    graph: Graph
    index: AclIndex
    model: ScriptedModel
    audit: InMemoryAuditLogger
    engine: PolicyGuardrailEngine = field(default_factory=PolicyGuardrailEngine)


def build_rig(
    docs: Sequence[IndexedDoc],
    reply: str | Callable[[Sequence[ChatMessage]], str] = "Torque is 140 Nm [1].",
    *,
    policy: GuardrailPolicy | None = None,
    store: ConversationStore | None = None,
) -> Rig:
    from ekip_agents import SYSTEM_PROMPT

    base = policy or GuardrailPolicy()
    base = (
        GuardrailPolicy.from_mapping(
            {
                "version": base.version,
                "max_input_chars": base.max_input_chars,
                "leak_markers": sorted(prompt_shingles(SYSTEM_PROMPT)),
            }
        )
        if policy is None
        else policy
    )
    engine = PolicyGuardrailEngine(base)
    audit = InMemoryAuditLogger()
    index, model = AclIndex(docs), ScriptedModel(reply)
    graph = build_agent_graph(
        router=RuleBasedRouter(),
        retriever=GuardedRetriever(index, engine, audit),
        model=GuardedChatModel(model, engine, audit),
        store=store,
    )
    return Rig(graph, index, model, audit, engine)
