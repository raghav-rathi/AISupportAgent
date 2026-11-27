-- Which process runs a session's turn, and whether that process is still alive. A claim is a lease:
-- the owner renews `turn_heartbeat_at` while the turn runs, and a claim that has not been renewed
-- for a while is free for the next process. Nothing has to clear claims when a process starts, so
-- a new instance cannot take a turn away from an old one that is still finishing it.
ALTER TABLE sessions
    ADD COLUMN turn_owner text,
    ADD COLUMN turn_heartbeat_at timestamptz,
    -- When the session last did anything. Old conversations are purged by this and not by their
    -- age, so a conversation in use is never deleted.
    ADD COLUMN last_active_at timestamptz NOT NULL DEFAULT now(),
    -- Keyed hashes of the PINs this conversation has used, right or wrong. They let a process that
    -- picks the session up later mask the same values again without storing them.
    ADD COLUMN secret_hashes text[] NOT NULL DEFAULT '{}';

UPDATE sessions SET last_active_at = greatest(created_at, coalesce(turn_started_at, created_at));
UPDATE sessions SET turn_running = false WHERE turn_running;

-- The events of one turn: its prompt traces on request, and what its earlier attempts used.
CREATE INDEX session_events_turn_idx ON session_events (session_id, turn_id);
-- What the deployment spent on model calls lately, for the daily budget.
CREATE INDEX session_events_spend_idx ON session_events (created_at)
    WHERE type IN ('llm.finished', 'retrieval.result');
CREATE INDEX worlds_created_idx ON worlds (created_at);
CREATE INDEX sessions_active_idx ON sessions (last_active_at);
