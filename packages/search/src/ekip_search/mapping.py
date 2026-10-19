"""Index name, settings and mapping. Pure functions: no I/O, so they are easy to test.

Design choices
- ONE shared index per embedding model version; every document carries `tenant_id` and every
  query MUST filter on it (and route by it). Per-tenant indices would not scale to thousands of
  small tenants; the tenant filter is covered by the security tests instead.
- The index name contains the model version: a new embedding model means a NEW index, built beside
  the old one and switched over, so vectors from different models can never be compared.
- `dynamic: strict` - an unexpected field is an error, not a silently guessed mapping.
- Vectors: HNSW on the Lucene engine with cosine similarity. Lucene applies the ACL/tenant filter
  *during* graph traversal (efficient filtering), so a restrictive filter does not return fewer
  than k results the way post-filtering would.
- Text analysis stays `standard` so exact identifiers ("AMM-32-11") still match as tokens.
"""

import re
from typing import Any

INDEX_PREFIX = "ekip-chunks"
MAPPING_VERSION = 1  # bump when the field layout changes in an incompatible way


def index_name_for(model_version: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", model_version.lower()).strip("-")
    return f"{INDEX_PREFIX}-{slug}"


def index_body(
    *, dimensions: int, model_version: str, shards: int = 1, replicas: int = 0, ef_search: int = 100
) -> dict[str, Any]:
    return {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": ef_search,
                "number_of_shards": shards,
                "number_of_replicas": replicas,
                "refresh_interval": "1s",
            }
        },
        "mappings": {
            "dynamic": "strict",
            "_meta": {
                "mapping_version": MAPPING_VERSION,
                "embedding_model_version": model_version,
                "dimensions": dimensions,
            },
            "properties": {
                # identity / scoping (filters, never analysed)
                "tenant_id": {"type": "keyword"},
                "document_id": {"type": "keyword"},
                "chunk_id": {"type": "keyword"},
                "document_version": {"type": "integer"},
                "ordinal": {"type": "integer"},
                "kind": {"type": "keyword"},
                "source": {"type": "keyword"},
                "mime_type": {"type": "keyword"},
                "source_uri": {"type": "keyword", "index": False, "doc_values": False},
                "content_hash": {"type": "keyword", "index": False},
                "asset_sha256": {"type": "keyword"},
                "page_number": {"type": "integer"},
                "last_modified": {"type": "date"},
                # access control
                "allowed_users": {"type": "keyword"},
                "allowed_groups": {"type": "keyword"},
                "acl_version": {"type": "keyword"},
                # lexical (BM25) fields; boosted separately at query time
                "title": {"type": "text"},
                "section_path": {"type": "text"},
                "content": {"type": "text"},
                # semantic
                "embedding_model_version": {"type": "keyword"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dimensions,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                        "parameters": {"m": 16, "ef_construction": 128},
                    },
                },
            },
        },
    }
