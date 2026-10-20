"""Document persistence adapters (ports live in ekip_contracts.document_store)."""

from ekip_persistence.conversations import InMemoryConversationStore, PostgresConversationStore
from ekip_persistence.dsn import to_asyncpg_dsn
from ekip_persistence.embedding_cache import RedisEmbeddingCache
from ekip_persistence.memory import InMemoryDocumentRepository
from ekip_persistence.object_store import InMemoryObjectStore, LocalFileObjectStore
from ekip_persistence.postgres import PostgresDocumentRepository, create_pool

__all__ = [
    "InMemoryConversationStore",
    "PostgresConversationStore",
    "InMemoryDocumentRepository",
    "InMemoryObjectStore",
    "LocalFileObjectStore",
    "PostgresDocumentRepository",
    "RedisEmbeddingCache",
    "create_pool",
    "to_asyncpg_dsn",
]
