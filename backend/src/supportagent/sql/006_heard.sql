-- What the customer had heard of the last reply when they spoke over it. Voice writes it at the
-- interruption, and the next turn puts it into the conversation in place of the full reply.
ALTER TABLE sessions ADD COLUMN heard text;
