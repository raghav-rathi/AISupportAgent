import { describe, expect, it } from "vitest";

import type { ServerEvent } from "./protocol";
import { type AppState, initialState, reduce, shownStatus } from "./state";

/** Feeds events to the reducer, numbering them as the server would. */
class Stream {
  state: AppState = initialState;
  private seq = 0;

  ready(sessionId = "s1", extra: Partial<Extract<ServerEvent, { type: "session.ready" }>> = {}): this {
    return this.raw({
      type: "session.ready",
      seq: 0,
      ts: 0,
      turn_id: null,
      session_id: sessionId,
      last_seq: 0,
      customer: null,
      tickets: [],
      busy: false,
      resumable: false,
      fault_injection: false,
      ...extra,
    });
  }

  /** One numbered event of turn `turn`. */
  send(turn: string, event: Record<string, unknown>): this {
    this.seq += 1;
    return this.raw({ seq: this.seq, ts: 100 + this.seq, turn_id: turn, ...event } as ServerEvent);
  }

  raw(event: ServerEvent): this {
    this.state = reduce(this.state, { type: "event", event });
    return this;
  }

  get turn() {
    return this.state.turns.at(-1)!;
  }
}

const started = (text = "Hi") => ({ type: "turn.started", mode: "text", text });
const delta = (text: string, call = "c1") => ({ type: "llm.delta", call_id: call, text });
const message = (text: string, held = false) => ({ type: "assistant.message", text, held, hits: [] });
const finished = (outcome = "answered") => ({
  type: "turn.finished",
  outcome,
  total_ms: 1200,
  llm_calls: 2,
  tool_calls: 0,
  input_tokens: 10,
  output_tokens: 5,
  cost_usd: 0.001,
});
const failed = { type: "turn.failed", code: "llm_unavailable", message: "quota", retryable: true, total_ms: 300 };

describe("a turn", () => {
  it("streams its answer and ends answered", () => {
    const s = new Stream().ready().send("t1", started()).send("t1", delta("Hel")).send("t1", delta("lo"));
    expect(s.turn.streaming).toBe("Hello");
    expect(s.state.busy).toBe(true);
    s.send("t1", message("Hello")).send("t1", finished());
    expect(s.turn.answer?.text).toBe("Hello");
    expect(s.turn.streaming).toBe("");
    expect([s.turn.status, s.state.busy]).toEqual(["answered", false]);
  });

  it("drops text that came with a tool call", () => {
    const s = new Stream()
      .ready()
      .send("t1", started())
      .send("t1", delta("Let me check that."))
      .send("t1", { type: "llm.reset", call_id: "c1", reason: "tool_call" });
    expect(s.turn.streaming).toBe("");
    // A reset for another call leaves the text of this one alone.
    s.send("t1", delta("Answer", "c2")).send("t1", { type: "llm.reset", call_id: "c1", reason: "timeout" });
    expect(s.turn.streaming).toBe("Answer");
  });

  it("keeps the hand-off and says why the first answer was withdrawn", () => {
    const ticket = { id: "ESC-1", reason: "ungrounded_answer", priority: "normal", summary: "s", account_number: null, tier: null };
    const s = new Stream()
      .ready()
      .send("t1", started())
      .send("t1", delta("Refunds take 365 days."))
      .send("t1", message("Refunds take 365 days."))
      .send("t1", { type: "answer.retracted", reason: "unsupported" });
    expect(s.turn.answer).toBeNull();
    s.send("t1", { type: "escalation.created", ticket }).send("t1", message("I've passed this on.", true)).send("t1", finished("escalated"));
    expect(s.turn.retracted).toBe("unsupported");
    expect(s.turn.answer).toMatchObject({ text: "I've passed this on.", held: true });
    expect(s.turn.tickets.map((t) => t.id)).toEqual(["ESC-1"]);
    expect(s.turn.status).toBe("escalated");
  });

  it("shows a tool's retries and then its result", () => {
    const s = new Stream()
      .ready()
      .send("t1", started())
      .send("t1", { type: "tool.started", tool_call_id: "k1", name: "search_knowledge_base", args: { query: "refunds" } })
      .send("t1", { type: "tool.error", tool_call_id: "k1", name: "search_knowledge_base", message: "timeout" })
      .send("t1", { type: "tool.retry", tool_call_id: "k1", name: "search_knowledge_base", attempt: 2, max_attempts: 3, wait_ms: 200 });
    expect(s.turn.steps[0]).toMatchObject({ status: "retrying", attempt: 2, maxAttempts: 3, error: "timeout" });
    s.send("t1", { type: "tool.finished", tool_call_id: "k1", name: "search_knowledge_base", outcome: "ok", summary: "5 passages", duration_ms: 480 });
    expect(s.turn.steps).toHaveLength(1);
    expect(s.turn.steps[0]).toMatchObject({ status: "ok", summary: "5 passages", durationMs: 480 });
  });

  it("can be picked up after it failed", () => {
    const s = new Stream().ready().send("t1", started()).send("t1", failed);
    expect([s.turn.status, s.state.busy, s.state.resumable]).toEqual(["failed", false, true]);
    // The retry carries the same turn ID and no second turn.started.
    s.send("t1", { type: "llm.started", call_id: "c2", purpose: "agent", model: "m", attempt: 1 });
    expect([s.turn.status, s.turn.failure, s.state.busy]).toEqual(["running", null, true]);
    s.send("t1", message("Done.")).send("t1", finished());
    expect(s.state.turns).toHaveLength(1);
    expect([s.turn.status, s.state.resumable]).toEqual(["answered", false]);
  });
});

describe("a connection", () => {
  it("ignores events it has already applied", () => {
    const s = new Stream().ready().send("t1", started()).send("t1", delta("Hello"));
    const before = s.state;
    // After a reconnect the server may send the tail again.
    s.raw({ seq: 2, ts: 102, turn_id: "t1", type: "llm.delta", call_id: "c1", text: "Hello" });
    expect(s.state).toBe(before);
    expect(s.ready("s1").turn.streaming).toBe("Hello"); // the same session keeps its conversation
  });

  it("starts over when the server answers with another session", () => {
    const s = new Stream().ready("s1").send("t1", started()).send("t1", message("Hello")).send("t1", finished());
    s.ready("s2", { fault_injection: true });
    expect(s.state.turns).toEqual([]);
    expect([s.state.sessionId, s.state.lastSeq, s.state.faultInjection]).toEqual(["s2", 0, true]);
  });

  it("does not mistake a replayed turn for one that is running", () => {
    // The server stopped while this turn ran, so the stored stream has no end for it.
    const s = new Stream().ready("s1", { last_seq: 2, busy: false, resumable: true });
    s.send("t1", started()).send("t1", { type: "llm.started", call_id: "c1", purpose: "agent", model: "m", attempt: 1 });
    expect([s.state.busy, s.state.resumable]).toEqual([false, true]);
    expect(shownStatus(s.state, s.turn)).toBe("interrupted");
    // What arrives after the replay is live again.
    s.send("t2", started("Again"));
    expect([s.state.busy, shownStatus(s.state, s.turn)]).toEqual([true, "running"]);
  });

  it("puts a refused message back in the user's hands", () => {
    const s = new Stream().ready();
    s.state = reduce(s.state, { type: "sent", text: "Hi" });
    expect(s.state.pending).toBe("Hi");
    s.raw({ seq: 0, ts: 0, turn_id: null, type: "error", code: "turn_in_progress", message: "wait for the current turn to finish", client_id: null });
    expect([s.state.pending, s.state.notice]).toEqual([null, "wait for the current turn to finish"]);
  });
});
