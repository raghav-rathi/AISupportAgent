CREATE TABLE sessions (
    id          text PRIMARY KEY,
    world_id    text NOT NULL REFERENCES worlds (id) ON DELETE CASCADE,
    created_at  timestamptz NOT NULL DEFAULT now()
);

-- The event stream of each session, without token deltas. The observability drawer reads it, and
-- so does a client that reconnects after the backend restarted.
CREATE TABLE session_events (
    session_id  text NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
    seq         bigint NOT NULL,
    turn_id     text,
    type        text NOT NULL,
    payload     jsonb NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, seq)
);
