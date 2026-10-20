"""conversation memory: question/answer turns per (tenant, user, conversation)

Revision ID: 0004
Revises: 0003
"""

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE conversation_turns (
            id              bigserial   PRIMARY KEY,
            tenant_id       text        NOT NULL CHECK (tenant_id ~ '^[A-Za-z0-9_-]{1,64}$'),
            user_subject    text        NOT NULL CHECK (length(user_subject) BETWEEN 1 AND 256),
            conversation_id text        NOT NULL CHECK (conversation_id ~ '^[A-Za-z0-9_-]{8,64}$'),
            question        text        NOT NULL CHECK (length(question) BETWEEN 1 AND 8000),
            answer          text        NOT NULL CHECK (length(answer) BETWEEN 1 AND 8000),
            created_at      timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX conversation_turns_thread ON conversation_turns "
        "(tenant_id, user_subject, conversation_id, id DESC)"
    )
    op.execute("CREATE INDEX conversation_turns_age ON conversation_turns (created_at)")


def downgrade() -> None:
    op.execute("DROP TABLE conversation_turns")
