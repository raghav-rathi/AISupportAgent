/**
 * The whole UI is a fold over the backend's event stream. Chat, step indicators, failure states
 * and the observability drawer all read the state this reducer builds, so a page reload that
 * replays the stored events shows the same conversation as the live stream did.
 */

import type {
  CaseUpdated,
  CustomerSummary,
  GuardrailVerdict,
  LlmFinished,
  LlmRetry,
  LlmTrace,
  PassageDetail,
  RetrievalHit,
  RetrievalResult,
  ServerEvent,
  SpeechHeard,
  SpeechSpoken,
  Ticket,
  TurnFinished,
  VoiceState,
} from "./protocol";

/**
 * `paused` waits for the visitor to come back before it connects again. `stopped` does not
 * connect again unless the visitor asks: the server said no, and asking again would not help.
 */
export type Connection = "connecting" | "open" | "reconnecting" | "paused" | "stopped";
export type FailureReason = LlmRetry["reason"];

export interface ToolStep {
  kind: "tool";
  id: string; // the tool call's ID
  name: string;
  args: Record<string, unknown>;
  /** `stopped`: the turn ended while this step was still running. */
  status: "running" | "retrying" | "ok" | "refused" | "failed" | "stopped";
  summary: string | null;
  error: string | null; // what the last failed attempt said
  attempt: number;
  maxAttempts: number | null;
  atMs: number; // when it started, on the turn's own clock
  durationMs: number | null;
  retrieval: RetrievalResult | null;
  receipt: Record<string, string> | null; // what an action did, as the tool reported it
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
  atMs: number; // when it started, on the turn's own clock. A call started before the turn starts at 0.
  attempts: number;
  finished: LlmFinished | null;
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
  clientId: string | null; // the ID this client gave the message, if it was sent from here
  userText: string;
  mode: "text" | "voice";
  startedTs: number;
  status: TurnStatus;
  phase: Phase;
  steps: Step[];
  calls: ModelCall[];
  streaming: string; // text of the answer in flight, not confirmed yet
  streamCallId: string | null;
  answer: Answer | null; // set once the answer has passed its check
  retracted: string | null; // why a streamed answer was taken back
  verdict: GuardrailVerdict | null;
  tickets: Ticket[];
  failure: { code: string; message: string; retryable: boolean } | null;
  resumable: boolean | null; // for a stopped turn: whether the server can pick it up
  finished: TurnFinished | null;
  totalMs: number | null;
  /**
   * What the customer waited for, in milliseconds from their message. Kept as events arrive and
   * replaced by the server's own figures when the turn ends, which a reload also has.
   */
  firstStepMs: number | null;
  firstTextMs: number | null;
  answerMs: number | null;
  /** Time the turn spent stopped, waiting to be picked up. It is not part of the turn's clock. */
  idleMs: number;
  stoppedTs: number | null;
  /** A spoken turn: how the customer's speech was taken, and each piece of the reply as audio. */
  heard: SpeechHeard | null;
  spoken: SpeechSpoken[];
  /** From the visitor's last word to the first sound of the reply, as this browser heard it. */
  heardAfterMs: number | null;
}

/** Voice in this page. The server decides what happens, and this is what it has said. */
export interface Voice {
  offered: boolean; // the server can listen and speak
  status: "off" | "starting" | "on";
  activity: VoiceState["state"] | null;
  partial: string; // the visitor's words so far, in the turn they are speaking
  bargeIn: boolean; // speaking over a reply stops it. Not without echo cancellation.
  notice: string | null; // why voice went off, when the visitor did not turn it off
}

/** The prompt traces of one turn, which the client asks for when it shows them. */
export interface TraceSet {
  status: "loading" | "ready";
  items: LlmTrace[];
  forCalls: number; // how many model calls the turn had finished when these were asked for
  /** Asked for while the turn ran. The server stores a turn's traces once the turn is over. */
  duringTurn: boolean;
}

export type Passage = PassageDetail | "loading" | "missing";

export interface AppState {
  connection: Connection;
  closeReason: string | null; // why the connection is down, when the server or the client said
  sessionId: string | null;
  lastSeq: number;
  /** The server is sending what this client missed. Those events rebuild turns and nothing else. */
  replaying: boolean;
  epoch: number; // counts the connections this page has had: a message belongs to one of them
  customer: CustomerSummary | null;
  tickets: Ticket[];
  /** What people at the service desk did to this conversation's tickets, as far as the customer is told. */
  updates: CaseUpdated[];
  turns: Turn[];
  busy: boolean; // a turn is running on the server
  resumable: boolean; // the last turn stopped early and can be picked up
  faultInjection: boolean; // the server lets this client simulate failures
  /** A message sent and not yet answered by the server, with the connection it was sent on. */
  pending: { id: string; text: string; epoch: number } | null;
  /** Text to put back into the composer: a message that turned out not to have been accepted. */
  draft: { text: string; n: number } | null;
  drafts: number;
  notice: string | null; // the server turned something down
  traces: Record<string, TraceSet>;
  passages: Record<string, Passage>;
  voice: Voice;
  /** What speech cost that belongs to no turn: a greeting, and listening while nobody spoke. */
  speechUsd: number;
  /** Counts the times voice went off without the visitor turning it off. */
  voiceDrops: number;
  /**
   * Model calls started early, for a turn the visitor seemed about to begin, that never came.
   * Null cost when one of them reported none.
   */
  unusedCalls: { count: number; costUsd: number | null };
}

const noVoice: Voice = { offered: false, status: "off", activity: null, partial: "", bargeIn: true, notice: null };

export const initialState: AppState = {
  connection: "connecting",
  closeReason: null,
  sessionId: null,
  lastSeq: 0,
  replaying: false,
  epoch: 0,
  customer: null,
  tickets: [],
  turns: [],
  busy: false,
  updates: [],
  resumable: false,
  faultInjection: false,
  pending: null,
  draft: null,
  drafts: 0,
  notice: null,
  traces: {},
  passages: {},
  voice: noVoice,
  speechUsd: 0,
  voiceDrops: 0,
  unusedCalls: { count: 0, costUsd: 0 },
};

export type Action =
  | { type: "event"; event: ServerEvent }
  | { type: "connection"; status: Connection; reason?: string }
  | { type: "sent"; id: string; text: string }
  | { type: "asked"; turnId: string }
  | { type: "looked"; chunkId: string }
  | { type: "reset" }
  | { type: "dismiss" }
  // Voice status moves only on what the voice link dispatches. It knows which start of voice an
  // event belongs to, and the events alone do not say.
  | { type: "voice"; status: Voice["status"]; notice?: string | null; bargeIn?: boolean }
  | { type: "voice-latency"; turnId: string; ms: number };

/** Hand a message that was not accepted back to the composer. */
function unsend(state: AppState): AppState {
  if (!state.pending) return state;
  const drafts = state.drafts + 1;
  return { ...state, pending: null, draft: { text: state.pending.text, n: drafts }, drafts };
}

export function reduce(state: AppState, action: Action): AppState {
  switch (action.type) {
    case "event":
      return applyEvent(state, action.event);
    case "connection": {
      const next = { ...state, connection: action.status, closeReason: action.reason || null };
      // Nothing will come back on a connection that is not being reopened, so nothing will say
      // whether the message arrived. The text goes back to the visitor.
      return action.status === "paused" || action.status === "stopped" ? unsend(next) : next;
    }
    case "sent":
      return { ...state, pending: { id: action.id, text: action.text, epoch: state.epoch }, notice: null };
    case "asked": {
      const turn = state.turns.find((t) => t.id === action.turnId);
      const before = state.traces[action.turnId];
      const forCalls = turn ? finishedCalls(turn) : 0;
      const duringTurn = turn?.status === "running";
      return {
        ...state,
        traces: { ...state.traces, [action.turnId]: { status: "loading", items: before?.items ?? [], forCalls, duringTurn } },
      };
    }
    case "looked":
      return typeof state.passages[action.chunkId] === "object"
        ? state
        : { ...state, passages: { ...state.passages, [action.chunkId]: "loading" } };
    case "reset": {
      // A message the old conversation never accepted goes back to the composer, not with it.
      const kept = unsend(state);
      return { ...initialState, draft: kept.draft, drafts: kept.drafts };
    }
    case "dismiss":
      return { ...state, notice: null, voice: { ...state.voice, notice: null } };
    case "voice":
      return {
        ...state,
        voice: {
          ...state.voice,
          status: action.status,
          activity: action.status === "on" ? "listening" : null,
          partial: "",
          bargeIn: action.bargeIn ?? state.voice.bargeIn,
          notice: action.notice ?? null,
        },
        voiceDrops: action.status === "off" && action.notice ? state.voiceDrops + 1 : state.voiceDrops,
      };
    case "voice-latency":
      return { ...state, turns: state.turns.map((turn) => (turn.id === action.turnId ? { ...turn, heardAfterMs: action.ms } : turn)) };
  }
}

function newTurn(id: string, text: string, mode: "text" | "voice", ts: number, clientId: string | null): Turn {
  return {
    id,
    clientId,
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
    resumable: null,
    finished: null,
    totalMs: null,
    firstStepMs: null,
    firstTextMs: null,
    answerMs: null,
    idleMs: 0,
    stoppedTs: null,
    heard: null,
    spoken: [],
    heardAfterMs: null,
  };
}

function withTicket(tickets: Ticket[], ticket: Ticket): Ticket[] {
  return tickets.some((t) => t.id === ticket.id) ? tickets : [...tickets, ticket];
}

function applyEvent(state: AppState, e: ServerEvent): AppState {
  if (e.type === "session.ready") {
    // A different ID means the server started a new session: the old one is gone.
    const same = state.sessionId === e.session_id;
    // What was asked on the connection that dropped will not be answered on this one. Taken out,
    // it is asked for again where it is shown.
    const kept = same
      ? {
          ...state,
          traces: Object.fromEntries(Object.entries(state.traces).filter(([, set]) => set.status !== "loading")),
          passages: Object.fromEntries(Object.entries(state.passages).filter(([, passage]) => passage !== "loading")),
        }
      : { ...initialState, pending: state.pending, draft: state.draft, drafts: state.drafts };
    return {
      ...kept,
      connection: "open",
      closeReason: null,
      sessionId: e.session_id,
      replaying: true,
      epoch: state.epoch + 1,
      customer: e.customer,
      tickets: e.tickets,
      busy: e.busy,
      resumable: e.resumable,
      faultInjection: e.fault_injection,
      // A new connection has no voice: the server's side of it went with the old one.
      voice: { ...noVoice, offered: e.voice, notice: same ? state.voice.notice : null },
    };
  }
  // The voice link acts on these for the start of voice they belong to, and tells the page.
  if (e.type === "voice.started" || e.type === "voice.stopped") return state;
  if (e.type === "voice.state") return { ...state, voice: { ...state.voice, activity: e.state } };
  if (e.type === "stt.partial") return { ...state, voice: { ...state.voice, partial: e.text } };
  if (e.type === "voice.playback") return state; // for the audio, not for the page
  if (e.type === "session.caught_up") {
    // Everything this client missed has arrived. A message sent on an earlier connection that
    // shows up in none of it never reached the server. Its text goes back to the visitor, who
    // decides whether to send it again: the client never sends a message twice by itself.
    const next = { ...state, replaying: false, turns: settled(state.turns, state.busy) };
    return next.pending && next.pending.epoch < next.epoch ? unsend(next) : next;
  }
  if (e.type === "error") {
    const next = { ...state, notice: e.message };
    if (e.code === "not_found") {
      // The answer names its passage. One from a server that names none is about every passage
      // still loading.
      const passages = e.chunk_id
        ? { ...state.passages, [e.chunk_id]: "missing" as const }
        : (Object.fromEntries(
            Object.entries(state.passages).map(([id, passage]) => [id, passage === "loading" ? "missing" : passage]),
          ) as Record<string, Passage>);
      return { ...next, notice: null, passages };
    }
    // The server answers every message under the ID the client gave it. This one started no turn.
    return e.client_id && state.pending?.id === e.client_id ? unsend(next) : next;
  }
  if (e.type === "trace.list") {
    const turn = state.turns.find((t) => t.id === e.turn_id);
    const held = state.traces[e.turn_id ?? ""];
    const forCalls = held?.forCalls ?? (turn ? finishedCalls(turn) : 0);
    const duringTurn = held?.duringTurn ?? turn?.status === "running";
    return e.turn_id
      ? { ...state, traces: { ...state.traces, [e.turn_id]: { status: "ready", items: e.traces, forCalls, duringTurn } } }
      : state;
  }
  if (e.type === "passage.detail") {
    return { ...state, passages: { ...state.passages, [e.chunk_id]: e } };
  }
  if (e.seq > 0) {
    if (e.seq <= state.lastSeq) return state; // seen before the connection dropped
    state = { ...state, lastSeq: e.seq };
  }
  if (e.type === "session.state") {
    // Which passages may be read is decided for whoever is verified now. With another account,
    // or none, the ones fetched before are asked for again.
    const same = (state.customer?.account_number ?? null) === (e.customer?.account_number ?? null);
    return { ...state, customer: e.customer, tickets: e.tickets, passages: same ? state.passages : {} };
  }
  if (e.type === "case.updated") {
    return { ...state, updates: [...state.updates, e] };
  }
  if (e.type === "voice.usage" || (e.type === "speech.spoken" && !e.turn_id)) {
    return { ...state, speechUsd: state.speechUsd + e.cost_usd };
  }
  if (e.type === "llm.finished" && !e.turn_id) {
    if (!e.unused) return state;
    const before = state.unusedCalls;
    const costUsd = before.costUsd == null || e.cost_usd == null ? null : before.costUsd + e.cost_usd;
    return { ...state, unusedCalls: { count: before.count + 1, costUsd } };
  }
  const live = !state.replaying;
  if (e.type === "turn.started") {
    const turn = newTurn(e.turn_id ?? `turn-${e.seq}`, e.text, e.mode, e.ts, e.client_id);
    const mine = e.client_id !== null && state.pending?.id === e.client_id;
    // What was said is now the turn's message, so the words in flight go.
    const voice = e.mode === "voice" ? { ...state.voice, partial: "" } : state.voice;
    const next = { ...state, turns: [...state.turns, turn], pending: mine ? null : state.pending, voice };
    return live ? { ...next, busy: true, resumable: false, notice: null } : next;
  }
  if (!e.turn_id) return state;

  let turns = state.turns;
  let index = turns.findLastIndex((turn) => turn.id === e.turn_id);
  if (index < 0) {
    turns = [...turns, newTurn(e.turn_id, "", "text", e.ts, null)];
    index = turns.length - 1;
  } else if (
    turns[index].status !== "running" &&
    !(turns[index].resumable && index === turns.length - 1) &&
    e.type !== "speech.heard" &&
    e.type !== "speech.spoken"
  ) {
    // The turn is over, and only the newest can be picked up again. What speech measured is
    // kept. Anything else that arrives late does not open the turn again.
    return state;
  }
  turns = turns.with(index, applyToTurn(turns[index], e));
  const tickets = e.type === "escalation.created" ? withTicket(state.tickets, e.ticket) : state.tickets;
  const next = { ...state, turns, tickets };
  if (!live) return next;
  switch (e.type) {
    case "turn.finished":
      return { ...next, busy: false, resumable: false };
    case "turn.failed":
      return { ...next, busy: false, resumable: e.retryable };
    case "turn.interrupted":
      return { ...next, busy: false, resumable: e.resumable };
    case "speech.heard":
    case "speech.spoken":
      return next; // speech is measured after its turn has ended: no turn is running
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

/**
 * After a replay, a turn that never got its last event and is not the one running stopped with
 * the server. A step it left running did not finish, and the server will not say so.
 */
function settled(turns: Turn[], busy: boolean): Turn[] {
  const stale = (turn: Turn, index: number) =>
    turn.status === "running" &&
    !(busy && index === turns.length - 1) &&
    turn.steps.some((step) => step.kind === "tool" && (step.status === "running" || step.status === "retrying"));
  if (!turns.some(stale)) return turns;
  return turns.map((turn, index) => (stale(turn, index) ? { ...turn, steps: stopSteps(turn.steps) } : turn));
}

/** A turn that ended while a step was still running: that step did not finish. */
function stopSteps(steps: Step[]): Step[] {
  return steps.map((step) =>
    step.kind === "tool" && (step.status === "running" || step.status === "retrying") ? { ...step, status: "stopped" } : step,
  );
}

function applyToTurn(before: Turn, e: ServerEvent): Turn {
  // What speech measured comes in after the turn has ended, and does not start it again.
  if (e.type === "speech.heard") return { ...before, heard: e };
  if (e.type === "speech.spoken") return { ...before, spoken: [...before.spoken, e] };
  let t = before;
  if (before.status === "failed" || before.status === "interrupted") {
    // An event after a stop means the turn was picked up again. What the stopped attempt left
    // on screen as its ending goes, and the time in between is not the turn's.
    const idle = before.stoppedTs == null ? 0 : Math.max(0, (e.ts - before.stoppedTs) * 1000);
    t = {
      ...before,
      status: "running",
      failure: null,
      resumable: null,
      totalMs: null,
      retracted: null,
      verdict: null,
      idleMs: before.idleMs + idle,
      stoppedTs: null,
    };
  }
  const at = Math.max(0, (e.ts - t.startedTs) * 1000 - t.idleMs);
  switch (e.type) {
    case "llm.started": {
      const fresh: ModelCall = { id: e.call_id, purpose: e.purpose, model: e.model, atMs: at, attempts: e.attempt, finished: null };
      const calls = withCall(t, e.call_id, (c) => ({ ...c, model: e.model, attempts: e.attempt }), fresh);
      return { ...t, calls, phase: e.purpose === "judge" ? "checking" : "thinking" };
    }
    case "llm.delta": {
      const first = t.streamCallId !== e.call_id;
      return {
        ...t,
        streaming: first ? e.text : t.streaming + e.text,
        streamCallId: e.call_id,
        phase: "writing",
        firstStepMs: t.firstStepMs ?? at,
        firstTextMs: first ? at : t.firstTextMs,
      };
    }
    case "llm.reset":
      // The text came with a tool call, the attempt died half way, or the turn was stopped.
      // Either way it is not the answer.
      return t.streamCallId === e.call_id
        ? { ...t, streaming: "", streamCallId: null, phase: "thinking", firstTextMs: null }
        : t;
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
        atMs: Math.max(0, at - e.total_ms),
        attempts: e.attempts,
        finished: null,
      };
      const calls = withCall(
        t,
        e.call_id,
        // A call started before its turn did is drawn from the turn's start.
        (c) => ({ ...c, model: e.model, attempts: e.attempts, finished: e, atMs: e.early_ms != null ? 0 : c.atMs }),
        fresh,
      );
      return { ...t, calls };
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
        atMs: at,
        durationMs: null,
        retrieval: null,
        receipt: null,
      };
      // A turn that was stopped inside a tool runs that same call again when it is picked up.
      // It is one step, not two.
      const again = t.steps.some((s) => s.kind === "tool" && s.id === e.tool_call_id);
      const steps = again ? withTool(t, e.tool_call_id, () => step) : [...t.steps, step];
      return { ...t, steps, phase: null, firstStepMs: t.firstStepMs ?? at };
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
          receipt: e.receipt,
        })),
      };
    case "retrieval.result":
      return { ...t, steps: withTool(t, e.tool_call_id, (s) => ({ ...s, retrieval: e })) };
    case "guardrail.verdict":
      return { ...t, verdict: e, phase: null };
    case "answer.retracted":
      return { ...t, retracted: e.reason, answer: null, streaming: "", streamCallId: null, firstTextMs: null };
    case "assistant.message":
      return {
        ...t,
        answer: { text: e.text, held: e.held, hits: e.hits },
        streaming: "",
        streamCallId: null,
        phase: null,
        firstTextMs: t.firstTextMs ?? at,
        answerMs: at,
      };
    case "escalation.created":
      return { ...t, tickets: withTicket(t.tickets, e.ticket) };
    case "turn.finished":
      return {
        ...t,
        status: e.outcome,
        finished: e,
        totalMs: e.total_ms,
        phase: null,
        streaming: "",
        streamCallId: null,
        // The server measured these at the source. A reload has them too, where the events that
        // carried the first text are no longer there to be timed.
        firstStepMs: e.first_step_ms ?? t.firstStepMs,
        firstTextMs: e.first_text_ms ?? t.firstTextMs,
        answerMs: e.answer_ms ?? t.answerMs,
      };
    case "turn.failed":
      return {
        ...t,
        status: "failed",
        failure: { code: e.code, message: e.message, retryable: e.retryable },
        resumable: e.retryable,
        totalMs: e.total_ms,
        phase: null,
        streaming: "",
        streamCallId: null,
        steps: stopSteps(t.steps),
        stoppedTs: e.ts,
      };
    case "turn.interrupted":
      return {
        ...t,
        status: "interrupted",
        resumable: e.resumable,
        totalMs: e.total_ms,
        phase: null,
        streaming: "",
        streamCallId: null,
        steps: stopSteps(t.steps),
        stoppedTs: e.ts,
      };
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

function finishedCalls(turn: Turn): number {
  return turn.calls.filter((call) => call.finished).length;
}

/**
 * True when the traces the client holds for this turn are older than the turn's model calls, or
 * were asked for while it ran and it has ended since. The server stores them at the end.
 */
export function tracesAreStale(state: AppState, turn: Turn): boolean {
  const held = state.traces[turn.id];
  if (!held) return true;
  return held.status === "ready" && (held.forCalls < finishedCalls(turn) || (held.duringTurn && turn.status !== "running"));
}

/** True when the drawer has something to show for an ended turn, a model call or not. */
export function hasTrace(turn: Turn): boolean {
  return turn.calls.length > 0 || turn.steps.length > 0 || turn.heard !== null || turn.spoken.length > 0;
}

export interface Usage {
  modelCalls: number;
  toolCalls: number;
  inputTokens: number;
  outputTokens: number;
  costUsd: number | null; // null when a model without a price took part
  speechUsd: number; // speech to text and text to speech, for a spoken turn
}

/** Models and speech together, or null when a part of it is unknown. */
export function totalCost(usage: Usage): number | null {
  return usage.costUsd == null ? null : usage.costUsd + usage.speechUsd;
}

function speechCost(turn: Turn): number {
  return (turn.heard?.cost_usd ?? 0) + turn.spoken.reduce((sum, piece) => sum + piece.cost_usd, 0);
}

/**
 * What the visitor heard of a reply that was cut off, or null when the reply was not cut off.
 * An empty string means none of it was heard. "One moment" is no reply.
 */
export function heardOf(turn: Turn): string | null {
  const last = turn.spoken.findLast((piece) => !piece.filler);
  return last?.interrupted && last.text_spoken != null ? last.text_spoken : null;
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
      speechUsd: speechCost(turn),
    };
  }
  const done = turn.calls.flatMap((call) => (call.finished ? [call.finished] : []));
  const searches = turn.steps.flatMap((step) => (step.kind === "tool" && step.retrieval ? [step.retrieval.cost_usd ?? 0] : []));
  return {
    modelCalls: done.length,
    toolCalls: turn.steps.filter((step) => step.kind === "tool").length,
    inputTokens: done.reduce((sum, call) => sum + (call.input_tokens ?? 0), 0),
    outputTokens: done.reduce((sum, call) => sum + (call.output_tokens ?? 0), 0),
    costUsd: done.some((call) => call.cost_usd == null)
      ? null
      : done.reduce((sum, call) => sum + (call.cost_usd ?? 0), 0) + searches.reduce((sum, cost) => sum + cost, 0),
    speechUsd: speechCost(turn),
  };
}

/** Where a finished call sits on its turn's clock: only the part inside the turn is drawn. */
export function callSpan(call: ModelCall): { startMs: number; durationMs: number } | null {
  const done = call.finished;
  if (!done) return null;
  const durationMs = done.early_ms == null ? done.total_ms : Math.max(0, done.total_ms - done.early_ms);
  return { startMs: call.atMs, durationMs };
}

/**
 * `otherSpeechUsd` is speech that belongs to no turn. `unusedModelUsd` is what model calls that
 * belong to no turn cost: they count toward the model cost.
 */
export function sessionUsage(
  turns: Turn[],
  otherSpeechUsd = 0,
  unusedModelUsd: number | null = 0,
): Usage & { turns: number; medianMs: number | null } {
  const each = turns.map(turnUsage);
  // A turn that failed or was stopped has a time too, and it says nothing about how long an
  // answer takes.
  const times = turns.flatMap((turn) => (turn.finished && turn.totalMs != null ? [turn.totalMs] : [])).sort((a, b) => a - b);
  return {
    turns: turns.length,
    modelCalls: each.reduce((sum, u) => sum + u.modelCalls, 0),
    toolCalls: each.reduce((sum, u) => sum + u.toolCalls, 0),
    inputTokens: each.reduce((sum, u) => sum + u.inputTokens, 0),
    outputTokens: each.reduce((sum, u) => sum + u.outputTokens, 0),
    costUsd:
      unusedModelUsd == null || each.some((u) => u.costUsd == null)
        ? null
        : each.reduce((sum, u) => sum + (u.costUsd ?? 0), unusedModelUsd),
    speechUsd: each.reduce((sum, u) => sum + u.speechUsd, otherSpeechUsd),
    medianMs: times.length ? times[Math.floor((times.length - 1) / 2)] : null,
  };
}
