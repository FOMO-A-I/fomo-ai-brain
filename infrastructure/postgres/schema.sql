-- Optional future schema for a multi-user website. Never execute against an
-- existing database without reviewing permissions, backups, and migrations.
CREATE TABLE IF NOT EXISTS fomo_conversations (
    id UUID PRIMARY KEY,
    owner_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS fomo_conversations_owner_idx
    ON fomo_conversations (owner_id, created_at DESC);

CREATE TABLE IF NOT EXISTS fomo_messages (
    id BIGSERIAL PRIMARY KEY,
    conversation_id UUID NOT NULL REFERENCES fomo_conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'tool')),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS fomo_messages_conversation_idx
    ON fomo_messages (conversation_id, id);

CREATE TABLE IF NOT EXISTS fomo_model_versions (
    version TEXT PRIMARY KEY,
    checkpoint_sha256 TEXT NOT NULL,
    dataset_manifest_sha256 TEXT NOT NULL,
    evaluation_report JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS fomo_audit_events (
    id BIGSERIAL PRIMARY KEY,
    owner_id TEXT,
    event_type TEXT NOT NULL,
    model_version TEXT,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);