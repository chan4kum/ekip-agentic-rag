"""image assets and chunk kind

Revision ID: 0002
Revises: 0001
"""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE chunks ADD COLUMN kind text NOT NULL DEFAULT 'text'"
        " CHECK (kind IN ('text','image'))"
    )
    op.execute(
        "ALTER TABLE chunks ADD COLUMN asset_sha256 text CHECK (asset_sha256 ~ '^[0-9a-f]{64}$')"
    )
    op.execute(
        "ALTER TABLE chunks ADD CONSTRAINT chunks_kind_asset_ck"
        " CHECK ((kind = 'image') = (asset_sha256 IS NOT NULL))"
    )
    op.execute(
        """
        CREATE TABLE image_assets (
            tenant_id        text    NOT NULL,
            document_id      text    NOT NULL,
            document_version integer NOT NULL,
            ordinal          integer NOT NULL CHECK (ordinal >= 0),
            sha256           text    NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
            mime_type        text    NOT NULL CHECK (mime_type ~ '^image/[a-z0-9][a-z0-9.+-]*$'),
            byte_size        integer NOT NULL CHECK (byte_size BETWEEN 1 AND 52428800),
            width            integer NOT NULL CHECK (width BETWEEN 1 AND 20000),
            height           integer NOT NULL CHECK (height BETWEEN 1 AND 20000),
            page_number      integer CHECK (page_number >= 1),
            bbox             double precision[] CHECK (array_length(bbox, 1) = 4),
            nearby_chunk_ordinal integer CHECK (nearby_chunk_ordinal >= 0),
            caption          text CHECK (length(caption) <= 5000),
            alt_text         text CHECK (length(alt_text) <= 5000),
            ocr_text         text CHECK (length(ocr_text) <= 5000),
            description      text CHECK (length(description) <= 5000),
            description_model_version text,
            PRIMARY KEY (tenant_id, document_id, document_version, ordinal),
            FOREIGN KEY (tenant_id, document_id, document_version)
                REFERENCES documents (tenant_id, document_id, version) ON DELETE CASCADE
        )
        """
    )
    op.execute("CREATE INDEX image_assets_sha_idx ON image_assets (tenant_id, sha256)")


def downgrade() -> None:
    op.execute("DROP TABLE image_assets")
    op.execute("ALTER TABLE chunks DROP CONSTRAINT chunks_kind_asset_ck")
    op.execute("ALTER TABLE chunks DROP COLUMN asset_sha256")
    op.execute("ALTER TABLE chunks DROP COLUMN kind")
