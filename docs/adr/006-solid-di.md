# ADR-006: SOLID, dependency injection, clean architecture

Status: Accepted · Date: 2026-10-19

## Rules
1. **Ports:** every external capability (search, cache, LLM, embeddings,
   object store, authorizer, tool client) is a `typing.Protocol` in the
   domain layer. Adapters implement it (OpenSearchRetriever, GroqLLM, ...).
2. **Constructor injection:** classes receive collaborators in `__init__`.
   No `import openai` / `redis.Redis()` inside business logic.
3. **One composition root per app** (the app factory). It is the only place
   that picks concrete adapters. FastAPI `Depends()` is used only at the
   HTTP edge to hand these to services.
4. **Small interfaces (ISP):** `ChunkReader` and `ChunkWriter` are separate ports.
5. **Substitutability (LSP):** every adapter passes the same contract test
   suite as its fake.
6. **Open/Closed:** a new provider = a new adapter class + one line in the
   composition root. No edits to services.
7. **No module-level singletons** or global clients in domain code.
8. Every service is unit-testable with fakes: no network, no Docker.

## Example
```python
class LLMClient(Protocol):                 # port
    async def complete(self, prompt: str) -> str: ...

class AnswerService:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
    async def answer(self, q: str) -> str:
        return await self._llm.complete(q)
```

## Consequences
+ Swap OpenSearch / Groq / Redis without touching agents or services
+ Fast, offline unit tests
- More small classes and files up front
