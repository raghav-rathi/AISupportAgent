/**
 * The whole UI is a fold over the backend's event stream. Chat, step indicators, failure states
 * and the observability drawer all read the state this reducer builds, so a page reload that
 * replays the stored events shows the same conversation as the live stream did.
 */

import type {
  CustomerSummary,
  GuardrailVerdict,
  LlmFinished,
  LlmRetry,
  LlmTrace,
  RetrievalHit,
  RetrievalResult,
  ServerEvent,
  Ticket,
  TurnFinished,
} from "./protocol";

export type Connection = "connecting" | "open" | "reconnecting";
export type FailureReason = LlmRetry["reason"];

export interface ToolStep {
  kind: "tool";
  id: string; // the tool call's ID
  name: string;
  args: Record<string, unknown>;
  status: "running" | "retrying" | "ok" | "refused" | "failed";
  summary: string | null;
  error: string | null; // what the last failed attempt said
  attempt: number;
  maxAttempts: number | null;
  startedTs: number;
  durationMs: number | null;
  retrieval: RetrievalResult | null;
}

/** The model failed once and the client is waiting to try again. */
export interface RetryNotice {
  kind: "retry";
  id: string;
  reason: FailureReason;
  model: string;
  attempt: number;
  maxAttempts: number;
  waitMs: number;
}

/** The first model gave up and the backup model took the call. */
export interface FallbackNotice {
  kind: "fallback";
  id: string;
  reason: FailureReason;
  fromModel: string;
  toModel: string;
}

export type Step = ToolStep | RetryNotice | FallbackNotice;

export interface ModelCall {
  id: string;
  purpose: "agent" | "judge";
  model: string;
  startedTs: number;
  attempts: number;
  finished: LlmFinished | null;
  trace: LlmTrace | null;
}

export type TurnStatus = "running" | "answered" | "escalated" | "failed" | "interrupted";
export type Phase = "thinking" | "writing" | "checking" | null;

export interface Answer {
  text: string;
  held: boolean; // it waited for its grounding check before going out
  hits: RetrievalHit[];
}

export interface Turn {
  id: string;
  userText: string;
  mode: "text" | "voice";
  startedTs: number;
  status: TurnStatus;
  phase: Phase;
  steps: Step[];
  calls: ModelCall[];
  streaming: string; // text of the answer in flight
  streamCallId: string | null;
  answer: Answer | null;
  retracted: string | null; // why a streamed answer was taken back
  verdict: GuardrailVerdict | null;
  tickets: Ticket[];
  failure: { code: string; message: string; retryable: boolean } | null;
  finished: TurnFinished | null;
  totalMs: number | null;
}

export interface AppState {
  connection: Connection;
  sessionId: string | null;
  lastSeq: number;
  /** Events up to this number are a replay of the past. They rebuild turns and nothing else. */
  replayUntil: number;
  customer: CustomerSummary | null;
  tickets: Ticket[];
  turns: Turn[];
  busy: boolean; // a turn is running on the server
  resumable: boolean; // the last turn stopped early and can be picked up
  faultInjection: boolean; // the server lets this client simulate failures
  pending: string | null; // a message sent and not yet confirmed by the server
  notice: string | null; // the server turned something down
}

export const initialState: AppState = {
  connection: "connecting",
  sessionId: null,
  lastSeq: 0,
  replayUntil: 0,
  customer: null,
  tickets: [],
  turns: [],
  busy: false,
  resumable: false,
  faultInjection: false,
  pending: null,
  notice: null,
};

export type Action =
  | { type: "event"; event: ServerEvent }
  | { type: "connection"; status: Connection }
  | { type: "sent"; text: string }
  | { type: "reset" }
  | { type: "dismiss" };

export function reduce(state: AppState, action: Action): AppState {
  switch (action.type) {
    case "event":
      return applyEvent(state, action.event);
    case "connection":
      return { ...state, connection: action.status };
    case "sent":
      return { ...state, pending: action.text, notice: null };
    case "reset":
      return { ...initialState };
    case "dismiss":
      return { ...state, notice: null };
  }
}

function newTurn(id: string, text: string, mode: "text" | "voice", ts: number): Turn {
  return {
    id,
    userText: text,
    mode,
    startedTs: ts,
    status: "running",
    phase: null,
    steps: [],
    calls: [],
    streaming: "",
    streamCallId: null,
    answer: null,
    retracted: null,
    verdict: null,
    tickets: [],
    failure: null,
    finished: null,
    totalMs: null,
  };
}

function applyEvent(state: AppState, e: ServerEvent): AppState {
  if (e.type === "session.ready") {
    // A different ID means the server started a new session: the old one is gone.
    const same = state.sessionId === e.session_id;
    const kept = same ? state : { ...initialState };
    return {
      ...kept,
      connection: "open",
      sessionId: e.session_id,
      replayUntil: e.last_seq,
      customer: e.customer,
      tickets: e.tickets,
      busy: e.busy,
      resumable: e.resumable,
      faultInjection: e.fault_injection,
    };
  }
  if (e.type === "error") {
    return { ...state, notice: e.message, pending: null };
  }
  if (e.seq > 0) {
    if (e.seq <= state.lastSeq) return state; // seen before the connection dropped
    state = { ...state, lastSeq: e.seq };
  }
  if (e.type === "session.state") {
    return { ...state, customer: e.customer, tickets: e.tickets };
  }
  const live = e.seq > state.replayUntil;
  if (e.type === "turn.started") {
    const turn = newTurn(e.turn_id ?? `turn-${e.seq}`, e.text, e.mode, e.ts);
    const next = { ...state, turns: [...state.turns, turn], pending: null };
    return live ? { ...next, busy: true, resumable: false, notice: null } : next;
  }
  if (!e.turn_id) return state;

  let turns = state.turns;
  let index = turns.findLastIndex((turn) => turn.id === e.turn_id);
  if (index < 0) {
    turns = [...turns, newTurn(e.turn_id, "", "text", e.ts)];
    index = turns.length - 1;
  }
  turns = turns.with(index, applyToTurn(turns[index], e));
  const next = { ...state, turns };
  if (!live) return next;
  switch (e.type) {
    case "turn.finished":
      return { ...next, busy: false, resumable: false };
    case "turn.failed":
      return { ...next, busy: false, resumable: e.retryable };
    case "turn.interrupted":
      return { ...next, busy: false, resumable: true };
    default:
      return { ...next, busy: true };
  }
}

function withCall(turn: Turn, id: string, change: (call: ModelCall) => ModelCall, fresh: ModelCall): ModelCall[] {
  return turn.calls.some((call) => call.id === id)
    ? turn.calls.map((call) => (call.id === id ? change(call) : call))
    : [...turn.calls, change(fresh)];
}

function withTool(turn: Turn, id: string, change: (step: ToolStep) => ToolStep): Step[] {
  return turn.steps.map((step) => (step.kind === "tool" && step.id === id ? change(step) : step));
}

function applyToTurn(before: Turn, e: ServerEvent): Turn {
  // An event after a failure means the turn was picked up again.
  const stopped = before.status === "failed" || before.status === "interrupted";
  const t: Turn = stopped ? { ...before, status: "running", failure: null, totalMs: null } : before;
  switch (e.type) {
    case "llm.started": {
      const fresh: ModelCall = {
        id: e.call_id,
        purpose: e.purpose,
        model: e.model,
        startedTs: e.ts,
        attempts: e.attempt,
        finished: null,
        trace: null,
      };
      const calls = withCall(t, e.call_id, (c) => ({ ...c, model: e.model, attempts: e.attempt }), fresh);
      return { ...t, calls, phase: e.purpose === "judge" ? "checking" : "thinking" };
    }
    case "llm.delta": {
      const text = t.streamCallId === e.call_id ? t.streaming + e.text : e.text;
      return { ...t, streaming: text, streamCallId: e.call_id, phase: "writing" };
    }
    case "llm.reset":
      // The text came with a tool call, or the attempt died half way. Either way it is not the answer.
      return t.streamCallId === e.call_id ? { ...t, streaming: "", streamCallId: null, phase: "thinking" } : t;
    case "llm.retry": {
      const notice: RetryNotice = {
        kind: "retry",
        id: `retry-${e.seq}`,
        reason: e.reason,
        model: e.model,
        attempt: e.attempt,
        maxAttempts: e.max_attempts,
        waitMs: e.wait_ms,
      };
      return { ...t, steps: [...t.steps, notice] };
    }
    case "llm.fallback": {
      const notice: FallbackNotice = {
        kind: "fallback",
        id: `fallback-${e.seq}`,
        reason: e.reason,
        fromModel: e.from_model,
        toModel: e.to_model,
      };
      return { ...t, steps: [...t.steps, notice] };
    }
    case "llm.finished": {
      const fresh: ModelCall = {
        id: e.call_id,
        purpose: e.purpose,
        model: e.model,
        startedTs: e.ts - e.total_ms / 1000,
        attempts: e.attempts,
        finished: null,
        trace: null,
      };
      const calls = withCall(t, e.call_id, (c) => ({ ...c, model: e.model, attempts: e.attempts, finished: e }), fresh);
      return { ...t, calls };
    }
    case "llm.trace": {
      const fresh: ModelCall = {
        id: e.call_id,
        purpose: e.purpose,
        model: e.model,
        startedTs: e.ts,
        attempts: 1,
        finished: null,
        trace: null,
      };
      return { ...t, calls: withCall(t, e.call_id, (c) => ({ ...c, trace: e }), fresh) };
    }
    case "tool.started": {
      const step: ToolStep = {
        kind: "tool",
        id: e.tool_call_id,
        name: e.name,
        args: e.args,
        status: "running",
        summary: null,
        error: null,
        attempt: 1,
        maxAttempts: null,
        startedTs: e.ts,
        durationMs: null,
        retrieval: null,
      };
      return { ...t, steps: [...t.steps, step], phase: null };
    }
    case "tool.error":
      return { ...t, steps: withTool(t, e.tool_call_id, (s) => ({ ...s, error: e.message })) };
    case "tool.retry":
      return {
        ...t,
        steps: withTool(t, e.tool_call_id, (s) => ({
          ...s,
          status: "retrying",
          attempt: e.attempt,
          maxAttempts: e.max_attempts,
        })),
      };
    case "tool.finished":
      return {
        ...t,
        steps: withTool(t, e.tool_call_id, (s) => ({
          ...s,
          status: e.outcome,
          summary: e.summary,
          durationMs: e.duration_ms,
        })),
      };
    case "retrieval.result":
      return { ...t, steps: withTool(t, e.tool_call_id, (s) => ({ ...s, retrieval: e })) };
    case "guardrail.verdict":
      return { ...t, verdict: e, phase: null };
    case "answer.retracted":
      return { ...t, retracted: e.reason, answer: null, streaming: "", streamCallId: null };
    case "assistant.message":
      return {
        ...t,
        answer: { text: e.text, held: e.held, hits: e.hits },
        streaming: "",
        streamCallId: null,
        phase: null,
      };
    case "escalation.created":
      return { ...t, tickets: [...t.tickets, e.ticket] };
    case "turn.finished":
      return { ...t, status: e.outcome, finished: e, totalMs: e.total_ms, phase: null, streaming: "", streamCallId: null };
    case "turn.failed":
      return {
        ...t,
        status: "failed",
        failure: { code: e.code, message: e.message, retryable: e.retryable },
        totalMs: e.total_ms,
        phase: null,
        streaming: "",
        streamCallId: null,
      };
    case "turn.interrupted":
      return { ...t, status: "interrupted", totalMs: e.total_ms, phase: null, streaming: "", streamCallId: null };
    default:
      return t;
  }
}

/**
 * What to show for a turn. A stored turn can lack its last event, when the server stopped while it
 * ran. Only the newest turn of a busy session is really running.
 */
export function shownStatus(state: AppState, turn: Turn): TurnStatus {
  if (turn.status !== "running") return turn.status;
  const newest = state.turns.at(-1)?.id === turn.id;
  return newest && state.busy ? "running" : "interrupted";
}

/** Milliseconds from the customer's message to the first token of the answer. */
export function firstTokenMs(turn: Turn): number | null {
  const answering = turn.calls.findLast((call) => call.purpose === "agent" && call.finished?.ttft_ms != null);
  if (!answering?.finished || answering.finished.ttft_ms == null) return null;
  return Math.max(0, (answering.startedTs - turn.startedTs) * 1000) + answering.finished.ttft_ms;
}

export interface Usage {
  modelCalls: number;
  toolCalls: number;
  inputTokens: number;
  outputTokens: number;
  costUsd: number | null; // null when a model without a price took part
}

/** Usage of one turn: the server's totals once it has ended, the calls seen so far before that. */
export function turnUsage(turn: Turn): Usage {
  if (turn.finished) {
    const f = turn.finished;
    return {
      modelCalls: f.llm_calls,
      toolCalls: f.tool_calls,
      inputTokens: f.input_tokens,
      outputTokens: f.output_tokens,
      costUsd: f.cost_usd,
    };
  }
  const done = turn.calls.flatMap((call) => (call.finished ? [call.finished] : []));
  return {
    modelCalls: done.length,
    toolCalls: turn.steps.filter((step) => step.kind === "tool").length,
    inputTokens: done.reduce((sum, call) => sum + (call.input_tokens ?? 0), 0),
    outputTokens: done.reduce((sum, call) => sum + (call.output_tokens ?? 0), 0),
    costUsd: done.some((call) => call.cost_usd == null) ? null : done.reduce((sum, call) => sum + (call.cost_usd ?? 0), 0),
  };
}

export function sessionUsage(turns: Turn[]): Usage & { turns: number; medianMs: number | null } {
  const each = turns.map(turnUsage);
  const times = turns.flatMap((turn) => (turn.totalMs == null ? [] : [turn.totalMs])).sort((a, b) => a - b);
  return {
    turns: turns.length,
    modelCalls: each.reduce((sum, u) => sum + u.modelCalls, 0),
    toolCalls: each.reduce((sum, u) => sum + u.toolCalls, 0),
    inputTokens: each.reduce((sum, u) => sum + u.inputTokens, 0),
    outputTokens: each.reduce((sum, u) => sum + u.outputTokens, 0),
    costUsd: each.some((u) => u.costUsd == null) ? null : each.reduce((sum, u) => sum + (u.costUsd ?? 0), 0),
    medianMs: times.length ? times[Math.floor((times.length - 1) / 2)] : null,
  };
}
