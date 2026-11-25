-- Refunds carry the calendar date the app's clock gave them, so a reply that names the date does
-- not depend on the database clock.
ALTER TABLE refunds ADD COLUMN refunded_on date;
UPDATE refunds SET refunded_on = created_at::date;
ALTER TABLE refunds ALTER COLUMN refunded_on SET NOT NULL;
ALTER TABLE refunds ADD CONSTRAINT refunds_amount_positive CHECK (amount_cents > 0);
ALTER TABLE refunds ADD CONSTRAINT refunds_customer_fk
    FOREIGN KEY (world_id, account_number)
    REFERENCES customers (world_id, account_number) ON DELETE CASCADE;

-- A ticket opened by a tool call is stored once, however often the call is retried.
ALTER TABLE tickets ADD COLUMN tool_call_id text;
CREATE UNIQUE INDEX tickets_tool_call_idx ON tickets (world_id, tool_call_id)
    WHERE tool_call_id IS NOT NULL;

-- Deleting a world cascades through these tables. Each needs an index on the side that points back.
CREATE INDEX sessions_world_idx ON sessions (world_id);
CREATE INDEX account_changes_account_idx ON account_changes (world_id, account_number);
CREATE INDEX refunds_invoice_idx ON refunds (world_id, invoice_id);
CREATE INDEX refunds_account_idx ON refunds (world_id, account_number);

-- A session's place in its event stream. Token deltas are numbered and not stored, so the highest
-- stored number is not the last number a client has seen.
ALTER TABLE sessions ADD COLUMN last_seq bigint NOT NULL DEFAULT 0;
UPDATE sessions s
SET last_seq = coalesce((SELECT max(seq) FROM session_events e WHERE e.session_id = s.id), 0);

-- The turn a session is running or ran last. `turn_running` keeps a second process from starting
-- a turn on the same session, and the ID and mode let a retry continue the turn it belongs to.
ALTER TABLE sessions
    ADD COLUMN turn_id text,
    ADD COLUMN turn_mode text NOT NULL DEFAULT 'text',
    ADD COLUMN turn_running boolean NOT NULL DEFAULT false,
    ADD COLUMN turn_started_at timestamptz;
