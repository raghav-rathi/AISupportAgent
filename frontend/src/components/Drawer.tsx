import { type KeyboardEvent, type ReactNode, useEffect, useRef, useState } from "react";

import { BAND_LABELS, TICKET_REASONS, TOOL_LABELS, formatCost, formatCount, formatMs, shortModel } from "@/lib/labels";
import type { FaultInject, LlmTrace, RetrievalResult, TraceMessage } from "@/lib/protocol";
import {
  type AppState,
  type ModelCall,
  type Passage,
  type Turn,
  type TurnStatus,
  callSpan,
  heardOf,
  sessionUsage,
  shownStatus,
  totalCost,
  tracesAreStale,
  turnUsage,
} from "@/lib/state";

import { Button, Caption, Glyph, Spinner, Tag } from "./ui";

type Tab = "turn" | "prompts" | "session" | "failures";

const STATUS_LABELS: Record<TurnStatus, string> = {
  running: "Running",
  answered: "Answered",
  escalated: "Handed to a person",
  failed: "Failed",
  interrupted: "Stopped",
};

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="flex flex-col gap-12">
      <h3 className="label-mono text-obsidian">{title}</h3>
      {children}
    </section>
  );
}

function Metric({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="flex min-w-0 flex-col gap-8 rounded-md bg-paper-white p-12 shadow-ring">
      <Caption>{label}</Caption>
      <p className="break-words text-heading font-regular tabular-nums text-obsidian">{value}</p>
      {note ? <p className="text-caption text-stone">{note}</p> : null}
    </div>
  );
}

function Quiet({ children }: { children: ReactNode }) {
  return <p className="text-caption text-stone">{children}</p>;
}

interface Bar {
  key: string;
  label: string;
  detail: string;
  startMs: number;
  durationMs: number;
  tone: string;
}

/** Every model call and tool call of the turn, placed on the turn's own clock. */
function timeline(turn: Turn): Bar[] {
  const bars: Bar[] = [];
  for (const call of turn.calls) {
    const span = callSpan(call);
    const done = call.finished;
    if (!span || !done) continue;
    const judge = call.purpose === "judge";
    const notes = [shortModel(call.model)];
    if (call.attempts > 1) notes.push(`${call.attempts} attempts`);
    // A call started before the turn shows only the part inside it.
    if (done.early_ms != null) notes.push(`started ${formatMs(done.early_ms)} early`);
    if (done.cancelled) notes.push("stopped");
    if (done.requests > 1) notes.push("asked twice");
    bars.push({
      key: call.id,
      label: judge ? "Grounding check" : "Model",
      detail: notes.join(", "),
      startMs: span.startMs,
      durationMs: span.durationMs,
      tone: done.cancelled ? "bg-ash" : judge ? "bg-charcoal" : "bg-obsidian",
    });
  }
  for (const step of turn.steps) {
    if (step.kind !== "tool" || step.durationMs == null) continue;
    bars.push({
      key: step.id,
      label: TOOL_LABELS[step.name]?.done ?? step.name,
      detail: step.status === "ok" ? "tool" : `tool, ${step.status}`,
      startMs: step.atMs,
      durationMs: step.durationMs,
      tone: step.status === "stopped" ? "bg-ash" : "bg-smoke",
    });
  }
  return bars.sort((a, b) => a.startMs - b.startMs);
}

function Timeline({ turn }: { turn: Turn }) {
  const bars = timeline(turn);
  if (bars.length === 0) return <Quiet>Nothing has finished yet.</Quiet>;
  // The clock leaves out the time a failed turn waited to be picked up, so the bars of both
  // attempts sit next to each other. The scale follows the bars.
  const span = Math.max(...bars.map((bar) => bar.startMs + bar.durationMs), 1);
  return (
    <ol className="flex flex-col gap-12">
      {bars.map((bar) => {
        const left = Math.min(99, Math.max(0, (bar.startMs / span) * 100));
        const width = Math.min(100 - left, Math.max(1, (bar.durationMs / span) * 100));
        return (
          <li key={bar.key} className="grid grid-cols-[128px_1fr_56px] items-center gap-12">
            <div className="min-w-0">
              <p className="truncate text-caption text-obsidian">{bar.label}</p>
              <p className="truncate font-mono text-label text-stone" title={bar.detail}>
                {bar.detail}
              </p>
            </div>
            <div className="relative h-8 overflow-hidden rounded-sm bg-hairline">
              <span className={`absolute inset-y-0 rounded-sm ${bar.tone}`} style={{ left: `${left}%`, width: `${width}%` }} />
            </div>
            <p className="text-right font-mono text-label tabular-nums text-stone">{formatMs(bar.durationMs)}</p>
          </li>
        );
      })}
    </ol>
  );
}

function score(value: number | null): string {
  return value == null ? "none" : value.toFixed(2);
}

function Retrieval({ result, passages, onPassage }: { result: RetrievalResult; passages: Record<string, Passage>; onPassage: (chunkId: string) => void }) {
  const [open, setOpen] = useState<string | null>(null);
  // Only a search that answered handed its passages to the model, and only those can be read.
  const passedOn = result.band === "answer" || result.band === "check";
  return (
    <div className="flex flex-col gap-12 rounded-md bg-paper-white p-12 shadow-ring">
      <p className="break-words font-mono text-caption text-obsidian">“{result.query}”</p>
      <div className="flex flex-wrap items-center gap-x-12 gap-y-4">
        <Tag>{BAND_LABELS[result.band]}</Tag>
        <span className="font-mono text-label tabular-nums text-stone">top score {score(result.top_score)}</span>
        <span className="font-mono text-label tabular-nums text-stone">
          thresholds {result.low} and {result.high}
        </span>
        {result.plan_score != null ? <span className="font-mono text-label tabular-nums text-stone">other plans {score(result.plan_score)}</span> : null}
      </div>
      {result.hits.length > 0 ? (
        <ol className="flex flex-col">
          {result.hits.map((hit) => {
            const passage = passages[hit.chunk_id];
            const row = (
              <>
                <span className="font-mono text-label text-stone">{hit.n}</span>
                <span className="font-mono text-label tabular-nums text-obsidian">{hit.score.toFixed(2)}</span>
                <span className="break-words text-caption text-obsidian">
                  {hit.source} ({hit.tiers.join(", ")})
                </span>
              </>
            );
            return (
              <li key={hit.chunk_id} className="flex flex-col gap-4 border-b border-hairline last:border-b-0">
                {passedOn ? (
                  <button
                    type="button"
                    aria-expanded={open === hit.chunk_id}
                    onClick={() => {
                      setOpen((was) => (was === hit.chunk_id ? null : hit.chunk_id));
                      if (typeof passage !== "object") onPassage(hit.chunk_id);
                    }}
                    className="-mx-8 grid grid-cols-[20px_44px_1fr] items-baseline gap-8 rounded-md px-8 py-8 text-left hover:bg-hairline/70"
                  >
                    {row}
                  </button>
                ) : (
                  <div className="grid grid-cols-[20px_44px_1fr] items-baseline gap-8 py-8">{row}</div>
                )}
                {open === hit.chunk_id ? (
                  <p className="mb-8 max-h-240 overflow-auto whitespace-pre-wrap break-words rounded-md bg-pure-white p-12 text-caption text-charcoal shadow-ring">
                    {passage === undefined || passage === "loading" ? "Loading the passage…" : passage === "missing" ? "This passage is no longer available." : passage.content}
                  </p>
                ) : null}
              </li>
            );
          })}
        </ol>
      ) : null}
      {!passedOn ? (
        <p className="text-caption text-stone">
          {result.hits.length > 0 ? "These were candidates only. No passage was passed to the model." : "No passage was passed to the model."}
        </p>
      ) : null}
      <p className="font-mono text-label tabular-nums text-stone">
        embed {formatMs(result.embed_ms)} · search {formatMs(result.search_ms)} · rerank {formatMs(result.rerank_ms)}
        {result.cost_usd != null ? ` · ${formatCost(result.cost_usd)}` : ""}
      </p>
    </div>
  );
}

const END_OF_TURN: Record<string, string> = {
  model: "the recogniser judged the sentence finished",
  timeout: "ended by silence",
  manual: "ended by the app",
};

/** A spoken turn: where the time went between the last word and the first sound, and what speech cost. */
function VoiceSection({ turn }: { turn: Turn }) {
  const heard = turn.heard;
  // What was said first, which is "one moment" when a tool ran, and the answer that stood: an
  // answer taken back is followed by the one spoken in its place.
  const first = turn.spoken[0];
  const reply = turn.spoken.findLast((piece) => !piece.filler);
  const covered = first?.filler === true && reply !== undefined;
  const firstToken = turn.calls.find((call) => call.purpose === "agent" && call.finished)?.finished?.ttft_ms;
  const chars = turn.spoken.reduce((sum, piece) => sum + piece.chars, 0);
  const cut = heardOf(turn);
  return (
    <Section title="Voice">
      <div className="grid grid-cols-2 gap-12">
        <Metric
          label="End-of-turn wait"
          value={formatMs(heard?.end_of_turn_ms)}
          note={heard ? `${END_OF_TURN[heard.trigger] ?? heard.trigger}${heard.digit_mode ? ", waiting out pauses for digits" : ""}` : undefined}
        />
        <Metric label="Model first token" value={formatMs(firstToken)} note="of the first model call" />
        <Metric label="Speech first audio" value={formatMs(reply?.synthesis_ms)} note="from the text going out to the first audio back" />
        <Metric
          label="Voice to voice"
          value={formatMs(first?.voice_to_voice_ms)}
          note={covered ? "last word to the first audio of “one moment”, at the server" : "last word to first audio, at the server"}
        />
        <Metric
          label="Voice to voice, here"
          value={formatMs(turn.heardAfterMs)}
          note={turn.heardAfterMs == null ? "measured live in this browser only" : "last word to first sound from the speaker, as measured here"}
        />
        {covered ? (
          <Metric
            label="Voice to answer"
            value={formatMs(reply?.voice_to_voice_ms)}
            note="last word to the first audio of the answer, after the tool and its check, at the server"
          />
        ) : null}
        <Metric
          label="Speech cost"
          value={formatCost(turnUsage(turn).speechUsd)}
          note={`${formatCount(Math.round(heard?.listened_s ?? 0))} s heard, ${formatCount(chars)} characters spoken`}
        />
      </div>
      {cut !== null ? (
        <Quiet>{cut ? `The reply was cut off. The customer heard: “${cut}”` : "The reply was cut off before any of it was heard."}</Quiet>
      ) : null}
    </Section>
  );
}

function TurnPanel({ turn, status, passages, onPassage }: { turn: Turn; status: TurnStatus; passages: Record<string, Passage>; onPassage: (chunkId: string) => void }) {
  const usage = turnUsage(turn);
  const searches = turn.steps.flatMap((step) => (step.kind === "tool" && step.retrieval ? [step.retrieval] : []));
  const verdict = turn.verdict;
  const held = turn.answer?.held ?? false;
  return (
    <>
      <div className="grid grid-cols-2 gap-12">
        <Metric label="End to end" value={formatMs(turn.totalMs)} note={STATUS_LABELS[status]} />
        <Metric label="Approved answer" value={formatMs(turn.answerMs)} note="complete, after its grounding check" />
        <Metric label="First sign of work" value={formatMs(turn.firstStepMs)} note="a step or the first words" />
        <Metric
          label="First text"
          value={formatMs(turn.firstTextMs)}
          note={held ? "held until the check had passed" : "of the answer that stood"}
        />
        <Metric
          label="Tokens"
          value={formatCount(usage.inputTokens + usage.outputTokens)}
          note={`${formatCount(usage.inputTokens)} in, ${formatCount(usage.outputTokens)} out`}
        />
        <Metric
          label="Estimated cost"
          value={formatCost(totalCost(usage))}
          note={`${usage.modelCalls} model calls, ${usage.toolCalls} tool calls${usage.speechUsd > 0 ? `, speech ${formatCost(usage.speechUsd)}` : ""}`}
        />
      </div>
      <Quiet>All four times count from the customer’s message. Time a stopped turn waited to be picked up is left out.</Quiet>

      {turn.mode === "voice" ? <VoiceSection turn={turn} /> : null}

      <Section title="Timeline">
        <Timeline turn={turn} />
      </Section>

      <Section title="Retrieval">
        {searches.length > 0 ? (
          searches.map((result) => <Retrieval key={result.seq} result={result} passages={passages} onPassage={onPassage} />)
        ) : (
          <Quiet>This turn did not search the knowledge base.</Quiet>
        )}
      </Section>

      <Section title="Grounding check">
        {verdict ? (
          <div className="flex flex-col gap-12 rounded-md bg-paper-white p-12 shadow-ring">
            <div className="flex flex-wrap gap-8">
              <Tag>{verdict.verdict}</Tag>
              <Tag>{verdict.held ? "answer held until checked" : "answer streamed, then checked"}</Tag>
              <Tag>{formatMs(verdict.ms)}</Tag>
            </div>
            {verdict.unsupported.map((claim, i) => (
              <p key={i} className="break-words text-caption text-charcoal">
                Unsupported: “{claim}”
              </p>
            ))}
            {turn.retracted ? <p className="text-caption text-stone">The answer was withdrawn and never stored.</p> : null}
          </div>
        ) : (
          <Quiet>No answer has been checked in this turn.</Quiet>
        )}
      </Section>
    </>
  );
}

function Message({ message, label }: { message: TraceMessage; label?: string }) {
  const calls = message.tool_calls.length > 0 ? JSON.stringify(message.tool_calls, null, 2) : null;
  const block =
    "max-h-240 overflow-auto whitespace-pre-wrap break-words rounded-md bg-paper-white p-12 shadow-ring font-mono text-label leading-[1.6] text-charcoal";
  return (
    <div className="flex flex-col gap-8">
      <Caption>
        {label ?? message.role}
        {message.name ? ` · ${message.name}` : ""}
      </Caption>
      {message.content ? <pre className={block}>{message.content}</pre> : null}
      {calls ? <pre className={block}>{calls}</pre> : null}
    </div>
  );
}

function Trace({ trace, call }: { trace: LlmTrace; call: ModelCall | undefined }) {
  const done = call?.finished;
  return (
    <details className="group rounded-md bg-pure-white shadow-card">
      <summary className="flex cursor-pointer list-none items-center gap-8 rounded-md p-12 [&::-webkit-details-marker]:hidden">
        <Glyph name="right" className="text-stone transition-transform group-open:rotate-90" />
        <span className="shrink-0 text-caption font-medium text-obsidian">{trace.purpose === "judge" ? "Grounding check" : "Agent"}</span>
        <span className="truncate font-mono text-label text-stone">{shortModel(trace.model)}</span>
        <span className="ml-auto shrink-0 font-mono text-label tabular-nums text-stone">
          {done
            ? done.cancelled
              ? `stopped · ${formatMs(done.total_ms)}`
              : `${formatCount(done.input_tokens)} in · ${formatCount(done.output_tokens)} out · ${formatMs(done.total_ms)}`
            : ""}
        </span>
      </summary>
      <div className="flex flex-col gap-12 px-12 pb-12">
        {done ? (
          <div className="flex flex-wrap gap-8">
            <Tag>first token {formatMs(done.ttft_ms)}</Tag>
            <Tag>{done.cancelled ? "cost not reported" : formatCost(done.cost_usd)}</Tag>
            {done.early_ms != null ? <Tag>started {formatMs(done.early_ms)} early</Tag> : null}
            {done.requests > 1 ? <Tag>asked twice, the first was slow to start</Tag> : null}
            {done.attempts > 1 ? <Tag>{done.attempts} attempts</Tag> : null}
            {done.used_fallback ? <Tag>backup model</Tag> : null}
          </div>
        ) : null}
        {trace.tools.length > 0 ? (
          <div className="flex flex-col gap-8">
            <Caption>Tools offered</Caption>
            <p className="break-words font-mono text-label leading-[1.6] text-stone">{trace.tools.join(", ")}</p>
          </div>
        ) : null}
        {trace.prompt.map((message, i) => (
          <Message key={i} message={message} />
        ))}
        <Message message={trace.response} label="response" />
      </div>
    </details>
  );
}

function PromptsPanel({ state, turn, onTraces }: { state: AppState; turn: Turn; onTraces: (turnId: string) => void }) {
  const held = state.traces[turn.id];
  const stale = tracesAreStale(state, turn);
  const online = state.connection === "open";
  // Traces are stored on the server and sent when someone looks. They are asked for when this
  // panel opens, and again when the turn has made a model call since.
  useEffect(() => {
    if (stale && online) onTraces(turn.id);
  }, [stale, online, turn.id, onTraces]);

  if (!held || (held.status === "loading" && held.items.length === 0)) {
    return (
      <p className="flex items-center gap-8 text-caption text-stone" role="status">
        <Spinner /> Loading the prompts of this turn…
      </p>
    );
  }
  if (held.items.length === 0) {
    return <Quiet>No prompt has been recorded for this turn yet. PINs are masked before a prompt is stored.</Quiet>;
  }
  return (
    <>
      <Quiet>
        Every model call of this turn, in order, with the prompt as it was sent. PINs are masked. Text that did not pass the grounding
        check is left out, and so is text a model sent along with a tool call.
      </Quiet>
      <div className="flex flex-col gap-12">
        {held.items.map((trace) => (
          <Trace key={`${trace.call_id}-${trace.seq}`} trace={trace} call={turn.calls.find((call) => call.id === trace.call_id)} />
        ))}
      </div>
    </>
  );
}

function Pair({ term, children }: { term: string; children: ReactNode }) {
  return (
    <>
      <dt className="label-mono flex items-center border-b border-hairline py-8 text-stone">{term}</dt>
      <dd className="min-w-0 truncate border-b border-hairline py-8 text-right text-caption tabular-nums text-obsidian">{children}</dd>
    </>
  );
}

function SessionPanel({ state }: { state: AppState }) {
  const usage = sessionUsage(state.turns, state.speechUsd, state.unusedCalls.costUsd);
  const unused = state.unusedCalls;
  const customer = state.customer;
  return (
    <>
      <Section title="This conversation">
        <dl className="grid grid-cols-[auto_1fr] gap-x-16">
          <Pair term="Session">{state.sessionId ? `${state.sessionId.slice(0, 8)}…` : "none"}</Pair>
          <Pair term="Customer">{customer ? `${customer.contact_name}, ${customer.company_name}` : "Not verified"}</Pair>
          {customer ? (
            <Pair term="Plan and status">
              {customer.tier}, {customer.status.replace("_", " ")}
            </Pair>
          ) : null}
          <Pair term="Turns">{usage.turns}</Pair>
          <Pair term="Median answered turn">{formatMs(usage.medianMs)}</Pair>
          <Pair term="Model calls">{usage.modelCalls}</Pair>
          <Pair term="Tool calls">{usage.toolCalls}</Pair>
          <Pair term="Tokens">
            {formatCount(usage.inputTokens)} in, {formatCount(usage.outputTokens)} out
          </Pair>
          <Pair term="Estimated model cost">{formatCost(usage.costUsd)}</Pair>
          {usage.speechUsd > 0 ? <Pair term="Speech cost">{formatCost(usage.speechUsd)}</Pair> : null}
          {usage.speechUsd > 0 ? <Pair term="Together">{formatCost(totalCost(usage))}</Pair> : null}
        </dl>
      </Section>
      {unused.count > 0 ? (
        <Section title="Voice">
          <dl className="grid grid-cols-[auto_1fr] gap-x-16">
            <Pair term="Model calls started early and not used">
              {unused.count}, {formatCost(unused.costUsd)}
            </Pair>
          </dl>
          <Quiet>Started while you were still speaking, to answer sooner. Their cost is part of the model cost above.</Quiet>
        </Section>
      ) : null}
      <Section title="Tickets">
        {state.tickets.length > 0 ? (
          state.tickets.map((ticket) => (
            <div key={ticket.id} className="flex flex-col gap-8 rounded-md bg-paper-white p-12 shadow-ring">
              <p className="font-mono text-caption text-obsidian">{ticket.id}</p>
              <div className="flex flex-wrap gap-8">
                <Tag>{TICKET_REASONS[ticket.reason]}</Tag>
                <Tag>{ticket.priority} priority</Tag>
              </div>
              <p className="break-words text-caption text-charcoal">{ticket.summary}</p>
            </div>
          ))
        ) : (
          <Quiet>No ticket has been opened.</Quiet>
        )}
      </Section>
    </>
  );
}

const FAILURES: { kind: FaultInject["kind"]; tool?: string; title: string; effect: string }[] = [
  { kind: "llm_rate_limit", title: "Rate limit", effect: "The next model call is rate limited once. The assistant waits, then tries again." },
  { kind: "llm_timeout", title: "Timeout", effect: "The next model call times out once. The assistant tries again." },
  { kind: "llm_server_error", title: "Server error", effect: "The next model call returns an error once. The assistant tries again." },
  { kind: "llm_primary_down", title: "Main model down", effect: "The main model refuses the next call, and the backup model answers." },
  { kind: "llm_down", title: "Every model down", effect: "The next turn fails. Try again picks it up where it stopped." },
  { kind: "judge_down", title: "Grounding check down", effect: "The answer cannot be checked, so it is withheld and the turn fails. Try again reruns the check." },
  { kind: "tool_failure", tool: "search_knowledge_base", title: "Search failure", effect: "The next knowledge base search fails once and is retried." },
  { kind: "speech_down", title: "Speech service down", effect: "Voice drops out if it is on, or fails to come on next time. The conversation goes on in text." },
];

function FailuresPanel({
  turnCount,
  voiceDrops,
  queued,
  onQueued,
  onInject,
}: {
  turnCount: number;
  voiceDrops: number;
  queued: Record<string, number>;
  onQueued: Props["onQueued"];
  onInject: Props["onInject"];
}) {
  return (
    <>
      <Quiet>Queue a simulated failure for your next message. It runs through the same code path as a real one.</Quiet>
      <ul className="flex flex-col gap-12">
        {FAILURES.map((failure) => {
          // A failure of speech is used up when voice drops, which no turn counts.
          const at = failure.kind === "speech_down" ? voiceDrops : turnCount;
          const isQueued = queued[failure.kind] === at;
          return (
            <li key={failure.kind} className="flex items-center gap-12 rounded-md bg-paper-white p-12 shadow-ring">
              <div className="flex min-w-0 flex-1 flex-col gap-4">
                <p className="text-body-sm font-medium text-obsidian">{failure.title}</p>
                <p className="text-caption text-stone">{failure.effect}</p>
              </div>
              {/* Queued, it stays a button, so the focus has somewhere to stay. */}
              <Button
                variant={isQueued ? "surface" : "ghost"}
                size="sm"
                aria-label={`Queue ${failure.title}`}
                aria-pressed={isQueued}
                aria-disabled={isQueued || undefined}
                onClick={() => {
                  if (!isQueued && onInject(failure.kind, failure.tool)) onQueued(failure.kind, at);
                }}
                className={isQueued ? "cursor-default hover:bg-hairline" : ""}
              >
                {isQueued ? (
                  <>
                    <Glyph name="check" className="text-terminal-green" /> Queued
                  </>
                ) : (
                  "Queue"
                )}
              </Button>
            </li>
          );
        })}
      </ul>
    </>
  );
}

interface Props {
  state: AppState;
  turnId: string | null; // null follows the newest turn
  overlay: boolean; // the drawer covers the conversation, as on a phone
  queued: Record<string, number>; // which failures are queued, and at which turn count or drop of voice
  onQueued: (kind: string, at: number) => void;
  onSelectTurn: (turnId: string | null) => void;
  onClose: () => void;
  /** False when it could not be sent, and then nothing is queued. */
  onInject: (kind: FaultInject["kind"], tool?: string) => boolean;
  onTraces: (turnId: string) => void;
  onPassage: (chunkId: string) => void;
}

export function Drawer({ state, turnId, overlay, queued, onQueued, onSelectTurn, onClose, onInject, onTraces, onPassage }: Props) {
  const [tab, setTab] = useState<Tab>("turn");
  const closer = useRef<HTMLButtonElement>(null);
  const root = useRef<HTMLElement>(null);
  const turns = state.turns;
  const found = turnId ? turns.findIndex((turn) => turn.id === turnId) : -1;
  const index = found >= 0 ? found : turns.length - 1;
  const turn = turns[index];
  const tabs: { id: Tab; label: string }[] = [
    { id: "turn", label: "Turn" },
    { id: "prompts", label: "Prompts" },
    { id: "session", label: "Session" },
    ...(state.faultInjection ? [{ id: "failures" as const, label: "Failures" }] : []),
  ];
  const shown = tabs.some((t) => t.id === tab) ? tab : "turn";
  const perTurn = shown === "turn" || shown === "prompts";
  const go = (to: number) => onSelectTurn(to >= turns.length - 1 ? null : turns[to].id);

  // Opening the drawer moves the keyboard into it. Closing it gives the focus back to the button
  // that opened it, which the caller does.
  useEffect(() => {
    closer.current?.focus();
  }, []);

  // Covering the page, the drawer keeps the keyboard. Tab goes round inside it, and Escape closes
  // it wherever the focus is, also after the control that had it is gone.
  useEffect(() => {
    if (!overlay) return;
    const onKey = (event: WindowEventMap["keydown"]) => {
      const panel = root.current;
      if (!panel) return;
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
        return;
      }
      if (event.key !== "Tab") return;
      const stops = [...panel.querySelectorAll<HTMLElement>("a[href], button, summary, input, select, textarea, [tabindex]")].filter(
        (element) => element.tabIndex >= 0 && !element.matches(":disabled") && element.getClientRects().length > 0,
      );
      if (stops.length === 0) return;
      const first = stops[0];
      const last = stops[stops.length - 1];
      const at = document.activeElement;
      if (!panel.contains(at) || at === (event.shiftKey ? first : last)) {
        event.preventDefault();
        (event.shiftKey ? last : first).focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [overlay, onClose]);

  const onTabKey = (event: KeyboardEvent<HTMLButtonElement>) => {
    const at = tabs.findIndex((t) => t.id === shown);
    const to = event.key === "ArrowRight" ? at + 1 : event.key === "ArrowLeft" ? at - 1 : event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : null;
    if (to === null) return;
    event.preventDefault();
    const next = tabs[(to + tabs.length) % tabs.length];
    setTab(next.id);
    document.getElementById(`drawer-tab-${next.id}`)?.focus();
  };

  return (
    <>
      {overlay ? <div aria-hidden="true" onClick={onClose} className="fixed inset-0 z-10 bg-paper-white/80" /> : null}
      <aside
        ref={root}
        id="observability"
        role={overlay ? "dialog" : undefined}
        aria-modal={overlay ? true : undefined}
        aria-label="Observability"
        onKeyDown={(event) => {
          if (event.key === "Escape") {
            event.stopPropagation();
            onClose();
          }
        }}
        className="flex w-480 shrink-0 flex-col overflow-hidden bg-pure-white lg:border-l lg:border-hairline max-lg:fixed max-lg:inset-12 max-lg:z-20 max-lg:w-auto max-lg:rounded-md max-lg:shadow-card"
      >
        <div className="flex flex-col gap-12 p-20 pb-16 max-sm:p-16 max-sm:pb-12">
          <div className="flex items-center justify-between gap-12">
            <h2 className="text-body font-medium text-obsidian">Observability</h2>
            <Button ref={closer} variant="quiet" size="sm" onClick={onClose} aria-label="Close observability" className="w-28 px-0">
              <Glyph name="cross" />
            </Button>
          </div>
          <div role="tablist" aria-label="What to show" className="flex gap-2 rounded-md bg-paper-white p-2 shadow-ring">
            {tabs.map((t) => (
              <button
                key={t.id}
                id={`drawer-tab-${t.id}`}
                type="button"
                role="tab"
                aria-selected={shown === t.id}
                aria-controls="drawer-panel"
                tabIndex={shown === t.id ? 0 : -1}
                onClick={() => setTab(t.id)}
                onKeyDown={onTabKey}
                className={`h-28 flex-1 rounded-sm px-8 text-caption transition-colors ${
                  shown === t.id ? "bg-pure-white text-obsidian shadow-ring" : "text-stone hover:text-obsidian"
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>
          {perTurn && turn ? (
            <div className="flex items-center justify-between gap-12">
              <p className="min-w-0 truncate text-caption text-charcoal">
                Turn {index + 1} of {turns.length}: “{turn.userText}”
              </p>
              <div className="flex shrink-0 gap-4">
                <Button variant="quiet" size="sm" disabled={index <= 0} onClick={() => go(index - 1)} aria-label="Earlier turn" className="w-28 px-0">
                  <Glyph name="left" />
                </Button>
                <Button
                  variant="quiet"
                  size="sm"
                  disabled={index >= turns.length - 1}
                  onClick={() => go(index + 1)}
                  aria-label="Later turn"
                  className="w-28 px-0"
                >
                  <Glyph name="right" />
                </Button>
              </div>
            </div>
          ) : null}
        </div>

        <div
          id="drawer-panel"
          role="tabpanel"
          aria-labelledby={`drawer-tab-${shown}`}
          tabIndex={0}
          className="relative flex min-h-0 flex-1 flex-col gap-24 overflow-y-auto px-20 pb-20 max-sm:px-16 max-sm:pb-16"
        >
          {shown === "session" ? <SessionPanel state={state} /> : null}
          {shown === "failures" ? (
            <FailuresPanel turnCount={turns.length} voiceDrops={state.voiceDrops} queued={queued} onQueued={onQueued} onInject={onInject} />
          ) : null}
          {perTurn && !turn ? <Quiet>Send a message to see what the assistant does with it.</Quiet> : null}
          {shown === "turn" && turn ? (
            <TurnPanel turn={turn} status={shownStatus(state, turn)} passages={state.passages} onPassage={onPassage} />
          ) : null}
          {shown === "prompts" && turn ? <PromptsPanel state={state} turn={turn} onTraces={onTraces} /> : null}
        </div>
      </aside>
    </>
  );
}
