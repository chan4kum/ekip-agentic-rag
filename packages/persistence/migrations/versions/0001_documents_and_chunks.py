"""documents and chunks

Revision ID: 0001
Revises:
"""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

# Frozen copy: a migration must not change when application code changes.
STATUSES = "'pending','parsing','chunking','embedding','indexing','indexed','failed','deleted'"


def upgrade() -> None:
    op.execute(
        f"""
        CREATE TABLE documents (
            tenant_id          text        NOT NULL CHECK (tenant_id ~ '^[A-Za-z0-9_-]{{1,64}}$'),
            document_id        text        NOT NULL CHECK (document_id ~ '^doc_[0-9a-f]{{32}}$'),
            version            integer     NOT NULL CHECK (version >= 1),
            source             text        NOT NULL CHECK (source ~ '^[a-z][a-z0-9_]{{0,31}}$'),
            source_document_id text        NOT NULL CHECK (length(source_document_id) BETWEEN 1 AND 512),
            source_version     text        NOT NULL CHECK (length(source_version) BETWEEN 1 AND 128),
            content_hash       text        NOT NULL CHECK (content_hash ~ '^[0-9a-f]{{64}}$'),
            title              text        NOT NULL CHECK (length(title) BETWEEN 1 AND 512),
            mime_type          text        NOT NULL,
            source_uri         text        NOT NULL CHECK (length(source_uri) BETWEEN 1 AND 2048),
            acl_version        text        NOT NULL,
            allowed_users      text[]      NOT NULL DEFAULT '{{}}',
            allowed_groups     text[]      NOT NULL DEFAULT '{{}}',
            last_modified      timestamptz NOT NULL,
            status             text        NOT NULL CHECK (status IN ({STATUSES})),
            failure_reason     text        CHECK (length(failure_reason) <= 1000),
            created_at         timestamptz NOT NULL DEFAULT now(),
            updated_at         timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (tenant_id, document_id, version)
        )
        """
    )
    op.execute(
        "CREATE INDEX documents_source_idx ON documents (tenant_id, source, source_document_id)"
    )
    op.execute(
        "CREATE INDEX documents_in_flight_idx ON documents (status, updated_at)"
        " WHERE status NOT IN ('indexed','deleted')"
    )
    op.execute(
        """
        CREATE TABLE chunks (
            tenant_id        text    NOT NULL,
            document_id      text    NOT NULL,
            document_version integer NOT NULL,
            chunk_id         text    NOT NULL,
            ordinal          integer NOT NULL CHECK (ordinal >= 0),
            content          text    NOT NULL CHECK (length(content) BETWEEN 1 AND 20000),
            content_hash     text    NOT NULL CHECK (content_hash ~ '^[0-9a-f]{64}$'),
            section_path     text[]  NOT NULL DEFAULT '{}',
            page_number      integer CHECK (page_number >= 1),
            PRIMARY KEY (tenant_id, chunk_id),
            UNIQUE (tenant_id, document_id, document_version, ordinal),
            FOREIGN KEY (tenant_id, document_id, document_version)
                REFERENCES documents (tenant_id, document_id, version) ON DELETE CASCADE
        )
        """
    )
    op.execute("CREATE INDEX chunks_hash_idx ON chunks (tenant_id, content_hash)")


def downgrade() -> None:
    op.execute("DROP TABLE chunks")
    op.execute("DROP TABLE documents")
