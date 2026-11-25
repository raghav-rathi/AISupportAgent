CREATE EXTENSION IF NOT EXISTS vector;

-- Knowledge base -----------------------------------------------------------------------------

CREATE TABLE kb_chunks (
    id            text PRIMARY KEY,
    doc_id        text NOT NULL,
    doc_title     text NOT NULL,
    heading_path  text[] NOT NULL,
    ordinal       integer NOT NULL,
    tiers         text[] NOT NULL,
    kind          text NOT NULL CHECK (kind IN ('prose', 'procedure', 'table', 'mixed')),
    content       text NOT NULL,
    embed_text    text NOT NULL,
    content_hash  text NOT NULL,
    fts           tsvector GENERATED ALWAYS AS (to_tsvector('english', embed_text)) STORED,
    updated_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX kb_chunks_fts_idx ON kb_chunks USING gin (fts);
CREATE INDEX kb_chunks_doc_idx ON kb_chunks (doc_id);

-- One row per chunk and embedding model, so models can be compared side by side.
-- The column has no fixed dimension and no ANN index: a few hundred chunks are scanned exactly.
CREATE TABLE kb_embeddings (
    chunk_id      text NOT NULL REFERENCES kb_chunks (id) ON DELETE CASCADE,
    model         text NOT NULL,
    content_hash  text NOT NULL,
    embedding     vector NOT NULL,
    PRIMARY KEY (chunk_id, model)
);

-- Mock CRM ------------------------------------------------------------------------------------
-- Every session and every eval scenario gets its own world: a private copy of the customers and
-- invoices. One visitor's refund never changes what the next visitor sees.

CREATE TABLE worlds (
    id          text PRIMARY KEY,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE customers (
    world_id          text NOT NULL REFERENCES worlds (id) ON DELETE CASCADE,
    account_number    text NOT NULL CHECK (account_number ~ '^[0-9]{8}$'),
    pin_hash          text NOT NULL,
    company_name      text NOT NULL,
    contact_name      text NOT NULL,
    contact_email     text NOT NULL,
    contact_phone     text NOT NULL,
    billing_email     text NOT NULL,
    tier              text NOT NULL CHECK (tier IN ('free', 'pro', 'enterprise')),
    billing_cycle     text CHECK (billing_cycle IN ('monthly', 'annual')),
    status            text NOT NULL CHECK (status IN ('active', 'past_due', 'suspended', 'cancelled')),
    seats             integer NOT NULL CHECK (seats > 0),
    marketing_opt_in  boolean NOT NULL DEFAULT false,
    customer_since    date NOT NULL,
    renews_on         date,
    PRIMARY KEY (world_id, account_number)
);

CREATE TABLE invoices (
    world_id        text NOT NULL,
    id              text NOT NULL,
    account_number  text NOT NULL,
    amount_cents    integer NOT NULL CHECK (amount_cents > 0),
    description     text NOT NULL,
    period_start    date NOT NULL,
    period_end      date NOT NULL,
    issued_on       date NOT NULL,
    status          text NOT NULL CHECK (status IN ('paid', 'open', 'refunded')),
    PRIMARY KEY (world_id, id),
    FOREIGN KEY (world_id, account_number)
        REFERENCES customers (world_id, account_number) ON DELETE CASCADE
);
CREATE INDEX invoices_account_idx ON invoices (world_id, account_number, issued_on DESC);

CREATE TABLE refunds (
    world_id         text NOT NULL,
    id               text NOT NULL,
    invoice_id       text NOT NULL,
    account_number   text NOT NULL,
    amount_cents     integer NOT NULL,
    reason           text NOT NULL,
    -- Customer plus invoice. A retry, or a second request for the same invoice, can never pay twice.
    idempotency_key  text NOT NULL,
    tool_call_id     text,
    created_at       timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (world_id, id),
    UNIQUE (world_id, idempotency_key),
    FOREIGN KEY (world_id, invoice_id) REFERENCES invoices (world_id, id) ON DELETE CASCADE
);

CREATE TABLE account_changes (
    id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    world_id        text NOT NULL,
    account_number  text NOT NULL,
    field           text NOT NULL,
    old_value       text,
    new_value       text,
    created_at      timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (world_id, account_number)
        REFERENCES customers (world_id, account_number) ON DELETE CASCADE
);

CREATE TABLE tickets (
    world_id            text NOT NULL REFERENCES worlds (id) ON DELETE CASCADE,
    id                  text NOT NULL,
    account_number      text,
    tier                text,
    reason              text NOT NULL,
    priority            text NOT NULL CHECK (priority IN ('low', 'normal', 'high', 'urgent')),
    summary             text NOT NULL,
    transcript_excerpt  text NOT NULL,
    status              text NOT NULL DEFAULT 'open',
    created_at          timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (world_id, id)
);
