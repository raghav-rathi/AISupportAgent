import { describe, expect, it } from "vitest";

import type { ServerEvent } from "./protocol";
import { type AppState, initialState, reduce, sessionUsage, shownStatus, tracesAreStale, turnUsage } from "./state";

/** Feeds events to the reducer, numbering them as the server would. */
class Stream {
  state: AppState = initialState;
  private seq = 0;

  /** A connection: the server's hello, and the marker that ends the replay of what was missed. */
  ready(sessionId = "s1", extra: Partial<Extract<ServerEvent, { type: "session.ready" }>> = {}): this {
    return this.hello(sessionId, extra).caughtUp();
  }

  hello(sessionId = "s1", extra: Partial<Extract<ServerEvent, { type: "session.ready" }>> = {}): this {
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

  caughtUp(): this {
    return this.raw({ type: "session.caught_up", seq: 0, ts: 0, turn_id: null });
  }

  /** One numbered event of turn `turn`, `at` seconds after the stream began. */
  send(turn: string, event: Record<string, unknown>, at?: number): this {
    this.seq += 1;
    return this.raw({ seq: this.seq, ts: at ?? 100 + this.seq, turn_id: turn, ...event } as ServerEvent);
  }

  raw(event: ServerEvent): this {
    this.state = reduce(this.state, { type: "event", event });
    return this;
  }

  say(text: string, id: string): this {
    this.state = reduce(this.state, { type: "sent", id, text });
    return this;
  }

  drop(status: "reconnecting" | "paused" | "stopped" = "reconnecting"): this {
    this.state = reduce(this.state, { type: "connection", status });
    return this;
  }

  refuse(code: string, clientId: string | null = null): this {
    return this.raw({ seq: 0, ts: 0, turn_id: null, type: "error", code, message: "no", client_id: clientId } as ServerEvent);
  }

  get turn() {
    return this.state.turns.at(-1)!;
  }

  /** Whether the composer would let the visitor send. */
  get canSend() {
    return this.state.connection === "open" && !this.state.busy && this.state.pending === null;
  }
}

const started = (text = "Hi", clientId: string | null = null) => ({ type: "turn.started", mode: "text", text, client_id: clientId });
const delta = (text: string, call = "c1") => ({ type: "llm.delta", call_id: call, text });
const message = (text: string, held = false) => ({ type: "assistant.message", text, held, hits: [] });
const model = (call = "c1", purpose = "agent") => ({ type: "llm.started", call_id: call, purpose, model: "m", attempt: 1 });
const modelDone = (call = "c1", totalMs = 500, purpose = "agent") => ({
  type: "llm.finished",
  call_id: call,
  purpose,
  model: "m",
  attempts: 1,
  used_fallback: false,
  ttft_ms: 100,
  total_ms: totalMs,
  input_tokens: 10,
  output_tokens: 5,
  cost_usd: 0.001,
});
const tool = (id = "k1", name = "issue_refund") => ({ type: "tool.started", tool_call_id: id, name, args: {} });
const toolDone = (id = "k1", name = "issue_refund", receipt: Record<string, string> | null = null) => ({
  type: "tool.finished",
  tool_call_id: id,
  name,
  outcome: "ok",
  summary: "done",
  duration_ms: 480,
  receipt,
});
const finished = (outcome = "answered", waits: Record<string, number | null> = {}) => ({
  type: "turn.finished",
  outcome,
  total_ms: 1200,
  llm_calls: 2,
  tool_calls: 0,
  input_tokens: 10,
  output_tokens: 5,
  cost_usd: 0.001,
  first_step_ms: null,
  first_text_ms: null,
  answer_ms: null,
  ...waits,
});
const failed = { type: "turn.failed", code: "llm_unavailable", message: "quota", retryable: true, total_ms: 300 };
const interrupted = (resumable = true) => ({ type: "turn.interrupted", total_ms: 300, resumable });
const ticket = (id: string) => ({ id, reason: "ungrounded_answer", priority: "normal", summary: "s", account_number: null, tier: null });

describe("a turn", () => {
  it("streams its answer and ends answered", () => {
    const s = new Stream().ready().send("t1", started()).send("t1", delta("Hel")).send("t1", delta("lo"));
    expect(s.turn.streaming).toBe("Hello");
    expect(s.turn.answer).toBeNull(); // text in flight is not the answer until its check has passed
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

  it("drops text that was still unconfirmed when the turn was stopped", () => {
    const s = new Stream()
      .ready()
      .send("t1", started())
      .send("t1", delta("Yes, 365 days."))
      .send("t1", { type: "llm.reset", call_id: "c1", reason: "cancelled" })
      .send("t1", interrupted());
    expect([s.turn.streaming, s.turn.answer, s.turn.status]).toEqual(["", null, "interrupted"]);
  });

  it("keeps the hand-off and says why the first answer was withdrawn", () => {
    const s = new Stream()
      .ready()
      .send("t1", started())
      .send("t1", delta("Refunds take 365 days."))
      .send("t1", { type: "guardrail.verdict", verdict: "ungrounded", held: false, unsupported: ["365 days"], ms: 900 })
      .send("t1", { type: "answer.retracted", reason: "unsupported" });
    expect([s.turn.answer, s.turn.streaming]).toEqual([null, ""]);
    s.send("t1", { type: "escalation.created", ticket: ticket("ESC-1") })
      .send("t1", message("I've passed this on.", true))
      .send("t1", finished("escalated"));
    expect(s.turn.retracted).toBe("unsupported");
    expect(s.turn.answer).toMatchObject({ text: "I've passed this on.", held: true });
    expect(s.turn.tickets.map((t) => t.id)).toEqual(["ESC-1"]);
    expect(s.turn.status).toBe("escalated");
  });

  it("shows a tool's retries and then its result", () => {
    const s = new Stream()
      .ready()
      .send("t1", started())
      .send("t1", tool("k1", "search_knowledge_base"))
      .send("t1", { type: "tool.error", tool_call_id: "k1", name: "search_knowledge_base", message: "timeout" })
      .send("t1", { type: "tool.retry", tool_call_id: "k1", name: "search_knowledge_base", attempt: 2, max_attempts: 3, wait_ms: 200 });
    expect(s.turn.steps[0]).toMatchObject({ status: "retrying", attempt: 2, maxAttempts: 3, error: "timeout" });
    s.send("t1", toolDone("k1", "search_knowledge_base"));
    expect(s.turn.steps).toHaveLength(1);
    expect(s.turn.steps[0]).toMatchObject({ status: "ok", summary: "done", durationMs: 480 });
  });

  it("keeps the receipt a refund came with", () => {
    const receipt = { kind: "refund", refund_id: "RF-1", invoice_id: "INV-30417", amount: "$96.00", date: "2026-11-24" };
    const s = new Stream().ready().send("t1", started()).send("t1", tool()).send("t1", toolDone("k1", "issue_refund", receipt));
    expect(s.turn.steps[0]).toMatchObject({ receipt });
  });

  it("can be picked up after it failed", () => {
    const s = new Stream().ready().send("t1", started()).send("t1", failed);
    expect([s.turn.status, s.state.busy, s.state.resumable]).toEqual(["failed", false, true]);
    // The retry carries the same turn ID and no second turn.started.
    s.send("t1", model("c2"));
    expect([s.turn.status, s.turn.failure, s.state.busy]).toEqual(["running", null, true]);
    s.send("t1", message("Done.")).send("t1", finished());
    expect(s.state.turns).toHaveLength(1);
    expect([s.turn.status, s.state.resumable]).toEqual(["answered", false]);
  });

  it("stops the spinner of a step the turn ended in, and runs it as one step when resumed", () => {
    const s = new Stream().ready().send("t1", started()).send("t1", tool()).send("t1", interrupted());
    expect(s.turn.steps[0]).toMatchObject({ status: "stopped" });
    // The resumed turn runs the same tool call again, under the same ID.
    s.send("t1", tool()).send("t1", toolDone());
    expect(s.turn.steps.map((step) => step.id)).toEqual(["k1"]);
    expect(s.turn.steps[0]).toMatchObject({ status: "ok" });
    expect(turnUsage(s.turn).toolCalls).toBe(1);
  });

  it("forgets the withdrawn answer of an attempt once the turn is picked up again", () => {
    const s = new Stream()
      .ready()
      .send("t1", started())
      .send("t1", delta("Hello."))
      .send("t1", { type: "guardrail.verdict", verdict: "skipped", held: false, unsupported: [], ms: 5 })
      .send("t1", { type: "answer.retracted", reason: "the grounding check could not run" })
      .send("t1", failed);
    expect(s.turn.retracted).not.toBeNull();
    s.send("t1", model("j2", "judge"))
      .send("t1", { type: "guardrail.verdict", verdict: "grounded", held: false, unsupported: [], ms: 700 })
      .send("t1", message("Hello."))
      .send("t1", finished());
    expect([s.turn.retracted, s.turn.verdict?.verdict, s.turn.answer?.text]).toEqual([null, "grounded", "Hello."]);
  });

  it("says when a stopped turn left nothing to pick up", () => {
    const s = new Stream().ready().send("t1", started()).send("t1", interrupted(false));
    expect([s.turn.resumable, s.state.resumable]).toEqual([false, false]);
  });
});

describe("what the customer waited for", () => {
  it("is timed from the events and leaves out text that was taken back", () => {
    const s = new Stream()
      .ready()
      .send("t1", started(), 100)
      .send("t1", delta("Let me check."), 100.4)
      .send("t1", { type: "llm.reset", call_id: "c1", reason: "tool_call" }, 100.5)
      .send("t1", tool("k1", "search_knowledge_base"), 100.6)
      .send("t1", delta("Fourteen", "c2"), 102)
      .send("t1", delta(" days.", "c2"), 102.5);
    expect(s.turn.firstStepMs).toBeCloseTo(400);
    expect(s.turn.firstTextMs).toBeCloseTo(2000); // the text that stood, not the text that was dropped
    expect(s.turn.answerMs).toBeNull();
    s.send("t1", message("Fourteen days."), 103.5);
    expect(s.turn.answerMs).toBeCloseTo(3500);
  });

  it("counts a held answer from the moment it appears", () => {
    const s = new Stream().ready().send("t1", started(), 100).send("t1", message("Fourteen days.", true), 104);
    expect([s.turn.firstTextMs, s.turn.answerMs]).toEqual([4000, 4000]);
  });

  it("takes the server's figures once the turn has ended, which a reload has too", () => {
    const waits = { first_step_ms: 310, first_text_ms: 1900, answer_ms: 3300 };
    const reloaded = new Stream().hello("s1", { last_seq: 3 });
    reloaded.send("t1", started(), 100).send("t1", message("Fourteen days."), 103).send("t1", finished("answered", waits), 103.1).caughtUp();
    expect([reloaded.turn.firstStepMs, reloaded.turn.firstTextMs, reloaded.turn.answerMs]).toEqual([310, 1900, 3300]);
  });

  it("puts the calls of a resumed turn on one clock, without the time it sat stopped", () => {
    const s = new Stream()
      .ready()
      .send("t1", started(), 100)
      .send("t1", model("c1"), 100.2)
      .send("t1", failed, 101)
      // An hour passes before the visitor presses "Try again".
      .send("t1", model("c2"), 3701)
      .send("t1", modelDone("c2"), 3701.5);
    expect(s.turn.calls.map((call) => Math.round(call.atMs))).toEqual([200, 1000]);
  });

  it("takes the median over answered turns only", () => {
    const s = new Stream().ready().send("t1", started()).send("t1", failed).send("t2", started()).send("t2", finished());
    expect(sessionUsage(s.state.turns).medianMs).toBe(1200);
  });
});

describe("a message on its way", () => {
  it("is confirmed by the turn that carries its ID", () => {
    const s = new Stream().ready().say("Hi", "m-1");
    expect([s.state.pending?.text, s.canSend]).toEqual(["Hi", false]);
    // A turn from another tab does not confirm it.
    s.send("t0", started("Other tab", "x-9"));
    expect(s.state.pending?.id).toBe("m-1");
    s.send("t0", finished()).send("t1", started("Hi", "m-1"));
    expect([s.state.pending, s.turn.clientId]).toEqual([null, "m-1"]);
  });

  it("goes back into the visitor's hands when the server never got it", () => {
    // The socket died with the message in it. The same conversation comes back, idle, and
    // nothing in what was missed carries the message.
    const s = new Stream().ready().say("Refund INV-30417 please", "m-1").drop().ready();
    expect([s.state.pending, s.state.draft?.text]).toEqual([null, "Refund INV-30417 please"]);
    expect(s.canSend).toBe(true);
    expect(s.state.turns).toEqual([]); // and it was not sent a second time behind their back
  });

  it("is not handed back when the server did get it", () => {
    const s = new Stream().ready().say("Refund INV-30417 please", "m-1").drop();
    // The turn started on the server while this client was away. The replay brings it.
    s.hello("s1", { busy: true }).send("t1", started("Refund INV-30417 please", "m-1")).caughtUp();
    expect([s.state.pending, s.state.draft, s.state.turns.length]).toEqual([null, null, 1]);
  });

  it("is not handed back for being sent while the replay was still arriving", () => {
    const s = new Stream().ready().drop().hello().say("Hi", "m-2").caughtUp();
    expect([s.state.pending?.id, s.state.draft]).toEqual(["m-2", null]);
  });

  it("goes back when the server refuses it, and stays when the refusal is about something else", () => {
    const s = new Stream().ready().say("Hi", "m-1").refuse("nothing_to_cancel");
    expect(s.state.pending?.id).toBe("m-1");
    s.refuse("too_long", "m-1");
    expect([s.state.pending, s.state.draft?.text, s.state.notice]).toEqual([null, "Hi", "no"]);
  });

  it("goes back when the connection will not be reopened by itself", () => {
    for (const status of ["paused", "stopped"] as const) {
      const s = new Stream().ready().say("Hi", "m-1").drop(status);
      expect([s.state.pending, s.state.draft?.text]).toEqual([null, "Hi"]);
    }
  });

  it("numbers each text it hands back, so the same text twice is still two", () => {
    const s = new Stream().ready().say("Hi", "m-1").refuse("too_long", "m-1");
    const first = s.state.draft?.n;
    s.say("Hi", "m-2").refuse("too_long", "m-2");
    expect(s.state.draft?.n).not.toBe(first);
  });
});

describe("a connection", () => {
  it("ignores events it has already applied", () => {
    const s = new Stream().ready().send("t1", started()).send("t1", delta("Hello"));
    const before = s.state.turns;
    // After a reconnect the server may send the tail again.
    s.raw({ seq: 2, ts: 102, turn_id: "t1", type: "llm.delta", call_id: "c1", text: "Hello" });
    expect(s.state.turns).toBe(before);
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
    const s = new Stream().hello("s1", { last_seq: 2, busy: false, resumable: true });
    s.send("t1", started()).send("t1", model()).caughtUp();
    expect([s.state.busy, s.state.resumable]).toEqual([false, true]);
    expect(shownStatus(s.state, s.turn)).toBe("interrupted");
    // What arrives after the replay is live again.
    s.send("t2", started("Again"));
    expect([s.state.busy, shownStatus(s.state, s.turn)]).toEqual([true, "running"]);
  });

  it("lists a ticket once, whether it came live, in a replay or with the session", () => {
    const s = new Stream().ready().send("t1", started()).send("t1", { type: "escalation.created", ticket: ticket("ESC-1") });
    expect(s.state.tickets.map((t) => t.id)).toEqual(["ESC-1"]);
    s.send("t1", { type: "session.state", customer: null, tickets: [ticket("ESC-1")] });
    s.send("t1", { type: "escalation.created", ticket: ticket("ESC-2") });
    expect(s.state.tickets.map((t) => t.id)).toEqual(["ESC-1", "ESC-2"]);
  });
});

describe("what is fetched on request", () => {
  it("asks again for a turn's traces once the turn has made another model call", () => {
    const s = new Stream().ready().send("t1", started()).send("t1", model()).send("t1", modelDone());
    expect(tracesAreStale(s.state, s.turn)).toBe(true);
    s.state = reduce(s.state, { type: "asked", turnId: "t1" });
    expect([s.state.traces.t1.status, tracesAreStale(s.state, s.turn)]).toEqual(["loading", false]);
    s.raw({ type: "trace.list", seq: 0, ts: 0, turn_id: "t1", traces: [] });
    expect([s.state.traces.t1.status, tracesAreStale(s.state, s.turn)]).toEqual(["ready", false]);
    s.send("t1", model("c2")).send("t1", modelDone("c2"));
    expect(tracesAreStale(s.state, s.turn)).toBe(true);
  });

  it("keeps a passage once it has it, and says so when the server has none", () => {
    const s = new Stream().ready();
    s.state = reduce(s.state, { type: "looked", chunkId: "a" });
    s.state = reduce(s.state, { type: "looked", chunkId: "b" });
    s.raw({ type: "passage.detail", seq: 0, ts: 0, turn_id: null, chunk_id: "a", source: "Refunds", tiers: ["pro"], content: "14 days." });
    s.refuse("not_found");
    expect(s.state.passages.a).toMatchObject({ content: "14 days." });
    expect([s.state.passages.b, s.state.notice]).toEqual(["missing", null]);
    s.state = reduce(s.state, { type: "looked", chunkId: "a" });
    expect(s.state.passages.a).toMatchObject({ content: "14 days." }); // not reset to loading
  });
});
