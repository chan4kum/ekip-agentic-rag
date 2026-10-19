"""source-side timestamps for ordering out-of-order events

Revision ID: 0003
Revises: 0002
"""

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE documents ADD COLUMN acl_as_of timestamptz")
    op.execute("ALTER TABLE documents ADD COLUMN deleted_at timestamptz")


def downgrade() -> None:
    op.execute("ALTER TABLE documents DROP COLUMN deleted_at")
    op.execute("ALTER TABLE documents DROP COLUMN acl_as_of")
