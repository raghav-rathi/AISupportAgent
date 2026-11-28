-- The service desk: what happens to a ticket after the assistant has opened it.

-- A workspace groups the conversations of one browser, so that its desk shows the tickets of
-- all of them and of nobody else. The column holds a hash of the key the browser keeps.
ALTER TABLE sessions ADD COLUMN workspace_id text;
CREATE INDEX sessions_workspace_idx ON sessions (workspace_id) WHERE workspace_id IS NOT NULL;

-- A ticket's ID, ESC-1A2B3C, is unique within its world and is what the customer is told. The
-- desk names a ticket by a key of its own that says nothing about the conversation it came from.
ALTER TABLE tickets
    ADD COLUMN key text NOT NULL DEFAULT replace(gen_random_uuid()::text, '-', ''),
    ADD COLUMN assignee text,
    ADD COLUMN channel text NOT NULL DEFAULT 'text' CHECK (channel IN ('text', 'voice')),
    ADD COLUMN updated_at timestamptz NOT NULL DEFAULT now();
CREATE UNIQUE INDEX tickets_key_idx ON tickets (key);
-- open: with the support team. pending: waiting on the customer. resolved: done.
ALTER TABLE tickets
    ADD CONSTRAINT tickets_status_check CHECK (status IN ('open', 'pending', 'resolved'));

-- Everything that happened to a ticket, in order. `public` is what the customer may read: a
-- reply, a change of status, who picked it up. A note never is.
CREATE TABLE ticket_activity (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    world_id    text NOT NULL,
    ticket_id   text NOT NULL,
    kind        text NOT NULL
        CHECK (kind IN ('created', 'assigned', 'status', 'priority', 'note', 'reply', 'refund')),
    author      text NOT NULL,
    public      boolean NOT NULL DEFAULT false,
    body        text NOT NULL DEFAULT '',
    meta        jsonb NOT NULL DEFAULT '{}',
    created_at  timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (world_id, ticket_id) REFERENCES tickets (world_id, id) ON DELETE CASCADE
);
CREATE INDEX ticket_activity_ticket_idx ON ticket_activity (world_id, ticket_id, id);

-- Tickets from before this migration get the one line every ticket starts with.
INSERT INTO ticket_activity (world_id, ticket_id, kind, author, public, body, created_at)
SELECT world_id, id, 'created', 'Assistant', true, summary, created_at FROM tickets;
