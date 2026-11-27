-- Speech costs money too, and the daily budget counts it. The index that the budget's query uses
-- has to cover the events that carry those costs.
DROP INDEX session_events_spend_idx;
CREATE INDEX session_events_spend_idx ON session_events (created_at)
    WHERE type IN ('llm.finished', 'retrieval.result', 'speech.heard', 'speech.spoken', 'voice.usage');
