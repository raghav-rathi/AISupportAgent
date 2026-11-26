import { type ReactNode, useState } from "react";

import { BAND_LABELS, TICKET_REASONS, TOOL_LABELS, formatCost, formatCount, formatMs, shortModel } from "@/lib/labels";
import type { FaultInject, RetrievalResult, TraceMessage } from "@/lib/protocol";
import { type AppState, type ModelCall, type Turn, type TurnStatus, firstTokenMs, sessionUsage, shownStatus, turnUsage } from "@/lib/state";

import { Button, Caption, Glyph, Tag } from "./ui";

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
      <h3 className="text-body-sm font-w480">{title}</h3>
      {children}
    </section>
  );
}

function Metric({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="flex flex-col gap-8 rounded-cards bg-onyx-canvas p-16">
      <Caption>{label}</Caption>
      <p className="font-arcadiadisplay text-subheading font-w480 tabular-nums">{value}</p>
      {note ? <p className="text-caption leading-[1.4] text-ash-text">{note}</p> : null}
    </div>
  );
}

function Quiet({ children }: { children: ReactNode }) {
  return <p className="text-body-sm leading-[1.45] text-ash-text">{children}</p>;
}

interface Bar {
  key: string;
  label: string;
  detail: string;
  startMs: number;
  durationMs: number;
  tone: string;
}

/** Every model call and tool call of the turn, placed on the turn's clock. */
function timeline(turn: Turn): Bar[] {
  const bars: Bar[] = [];
  for (const call of turn.calls) {
    if (!call.finished) continue;
    const judge = call.purpose === "judge";
    bars.push({
      key: call.id,
      label: judge ? "Grounding check" : "Model",
      detail: shortModel(call.model) + (call.attempts > 1 ? `, ${call.attempts} attempts` : ""),
      startMs: (call.startedTs - turn.startedTs) * 1000,
      durationMs: call.finished.total_ms,
      tone: judge ? "bg-ash-text" : "bg-ivory-text",
    });
  }
  for (const step of turn.steps) {
    if (step.kind !== "tool" || step.durationMs == null) continue;
    bars.push({
      key: step.id,
      label: TOOL_LABELS[step.name]?.done ?? step.name,
      detail: step.status === "ok" ? "tool" : `tool, ${step.status}`,
      startMs: (step.startedTs - turn.startedTs) * 1000,
      durationMs: step.durationMs,
      tone: "bg-slate-border",
    });
  }
  return bars.sort((a, b) => a.startMs - b.startMs);
}

function Timeline({ turn }: { turn: Turn }) {
  const bars = timeline(turn);
  if (bars.length === 0) return <Quiet>Nothing has finished yet.</Quiet>;
  // A turn that failed and was picked up later has a gap in it, so the scale follows the bars.
  const span = Math.max(turn.totalMs ?? 0, ...bars.map((bar) => bar.startMs + bar.durationMs), 1);
  return (
    <ol className="flex flex-col gap-12">
      {bars.map((bar) => (
        <li key={bar.key} className="grid grid-cols-[128px_1fr_56px] items-center gap-12">
          <div className="min-w-0">
            <p className="truncate text-caption leading-[1.35]">{bar.label}</p>
            <p className="truncate text-caption leading-[1.35] text-ash-text">{bar.detail}</p>
          </div>
          <div className="relative h-8 rounded-full bg-onyx-canvas">
            <span
              className={`absolute inset-y-0 rounded-full ${bar.tone}`}
              style={{
                left: `${Math.min(99, (Math.max(0, bar.startMs) / span) * 100)}%`,
                width: `${Math.max(1, (bar.durationMs / span) * 100)}%`,
              }}
            />
          </div>
          <p className="text-right text-caption tabular-nums text-ash-text">{formatMs(bar.durationMs)}</p>
        </li>
      ))}
    </ol>
  );
}

function score(value: number | null): string {
  return value == null ? "none" : value.toFixed(2);
}

function Retrieval({ result }: { result: RetrievalResult }) {
  return (
    <div className="flex flex-col gap-12 rounded-cards bg-onyx-canvas p-16">
      <p className="text-body-sm leading-[1.45]">“{result.query}”</p>
      <div className="flex flex-wrap gap-8">
        <Tag>{BAND_LABELS[result.band]}</Tag>
        <Tag>top score {score(result.top_score)}</Tag>
        <Tag>
          thresholds {result.low} and {result.high}
        </Tag>
        {result.plan_score != null ? <Tag>other plans {score(result.plan_score)}</Tag> : null}
      </div>
      {result.hits.length > 0 ? (
        <ol className="flex flex-col gap-8">
          {result.hits.map((hit) => (
            <li key={hit.chunk_id} className="grid grid-cols-[20px_44px_1fr] gap-8 text-caption leading-[1.4]">
              <span className="text-ash-text">{hit.n}</span>
              <span className="tabular-nums">{hit.score.toFixed(2)}</span>
              <span className="text-ash-text">
                {hit.source} ({hit.tiers.join(", ")})
              </span>
            </li>
          ))}
        </ol>
      ) : (
        <p className="text-caption leading-[1.4] text-ash-text">No passage was passed to the model.</p>
      )}
      <p className="text-caption leading-[1.4] text-ash-text">
        embed {formatMs(result.embed_ms)} · search {formatMs(result.search_ms)} · rerank {formatMs(result.rerank_ms)}
      </p>
    </div>
  );
}

function TurnPanel({ turn, status }: { turn: Turn; status: TurnStatus }) {
  const usage = turnUsage(turn);
  const searches = turn.steps.flatMap((step) => (step.kind === "tool" && step.retrieval ? [step.retrieval] : []));
  const verdict = turn.verdict;
  return (
    <>
      <div className="grid grid-cols-2 gap-12">
        <Metric label="End to end" value={formatMs(turn.totalMs)} note={STATUS_LABELS[status]} />
        <Metric label="First token" value={formatMs(firstTokenMs(turn))} note="of the final answer" />
        <Metric
          label="Tokens"
          value={formatCount(usage.inputTokens + usage.outputTokens)}
          note={`${formatCount(usage.inputTokens)} in, ${formatCount(usage.outputTokens)} out`}
        />
        <Metric
          label="Estimated cost"
          value={formatCost(usage.costUsd)}
          note={`${usage.modelCalls} model calls, ${usage.toolCalls} tool calls`}
        />
      </div>

      <Section title="Timeline">
        <Timeline turn={turn} />
      </Section>

      <Section title="Retrieval">
        {searches.length > 0 ? (
          searches.map((result) => <Retrieval key={result.seq} result={result} />)
        ) : (
          <Quiet>This turn did not search the knowledge base.</Quiet>
        )}
      </Section>

      <Section title="Grounding check">
        {verdict ? (
          <div className="flex flex-col gap-12 rounded-cards bg-onyx-canvas p-16">
            <div className="flex flex-wrap gap-8">
              <Tag>{verdict.verdict}</Tag>
              <Tag>{verdict.held ? "answer held until checked" : "answer streamed, then checked"}</Tag>
              <Tag>{formatMs(verdict.ms)}</Tag>
            </div>
            {verdict.unsupported.map((claim, i) => (
              <p key={i} className="text-caption leading-[1.4] text-ash-text">
                Unsupported: “{claim}”
              </p>
            ))}
            {turn.retracted ? <p className="text-caption leading-[1.4] text-ash-text">Withdrawn: {turn.retracted}</p> : null}
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
  const block = "max-h-240 overflow-auto whitespace-pre-wrap break-words rounded-lg bg-graphite-card p-12 font-mono text-caption leading-[1.5]";
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

function Trace({ call }: { call: ModelCall }) {
  const trace = call.trace;
  if (!trace) return null;
  const done = call.finished;
  return (
    <details className="group rounded-cards bg-onyx-canvas">
      <summary className="flex cursor-pointer list-none items-center gap-8 rounded-cards p-16 text-body-sm [&::-webkit-details-marker]:hidden">
        <Glyph name="right" className="transition-transform group-open:rotate-90" />
        <span className="font-w480">{call.purpose === "judge" ? "Grounding check" : "Agent"}</span>
        <span className="truncate text-ash-text">{shortModel(call.model)}</span>
        <span className="ml-auto shrink-0 text-caption tabular-nums text-ash-text">
          {done ? `${formatCount(done.input_tokens)} in · ${formatCount(done.output_tokens)} out · ${formatMs(done.total_ms)}` : ""}
        </span>
      </summary>
      <div className="flex flex-col gap-16 px-16 pb-16">
        {done ? (
          <div className="flex flex-wrap gap-8">
            <Tag>first token {formatMs(done.ttft_ms)}</Tag>
            <Tag>{formatCost(done.cost_usd)}</Tag>
            {done.attempts > 1 ? <Tag>{done.attempts} attempts</Tag> : null}
            {done.used_fallback ? <Tag>backup model</Tag> : null}
          </div>
        ) : null}
        {trace.tools.length > 0 ? (
          <div className="flex flex-col gap-8">
            <Caption>Tools offered</Caption>
            <p className="font-mono text-caption leading-[1.5] text-ash-text">{trace.tools.join(", ")}</p>
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

function PromptsPanel({ turn }: { turn: Turn }) {
  const traced = turn.calls.filter((call) => call.trace);
  if (traced.length === 0) {
    return <Quiet>No prompt has been recorded for this turn yet. PINs are masked before a prompt is stored.</Quiet>;
  }
  return (
    <>
      <Quiet>Every model call of this turn, in order, with the prompt as it was sent. PINs are masked.</Quiet>
      <div className="flex flex-col gap-12">
        {traced.map((call) => (
          <Trace key={call.id} call={call} />
        ))}
      </div>
    </>
  );
}

function Pair({ term, children }: { term: string; children: ReactNode }) {
  return (
    <>
      <dt className="text-ash-text">{term}</dt>
      <dd className="min-w-0 truncate text-right tabular-nums">{children}</dd>
    </>
  );
}

function SessionPanel({ state }: { state: AppState }) {
  const usage = sessionUsage(state.turns);
  const customer = state.customer;
  return (
    <>
      <Section title="This conversation">
        <dl className="grid grid-cols-[auto_1fr] gap-x-16 gap-y-12 rounded-cards bg-onyx-canvas p-16 text-body-sm leading-[1.45]">
          <Pair term="Session">{state.sessionId ? `${state.sessionId.slice(0, 8)}…` : "none"}</Pair>
          <Pair term="Customer">{customer ? `${customer.contact_name}, ${customer.company_name}` : "Not verified"}</Pair>
          {customer ? (
            <Pair term="Plan and status">
              {customer.tier}, {customer.status.replace("_", " ")}
            </Pair>
          ) : null}
          <Pair term="Turns">{usage.turns}</Pair>
          <Pair term="Median turn">{formatMs(usage.medianMs)}</Pair>
          <Pair term="Model calls">{usage.modelCalls}</Pair>
          <Pair term="Tool calls">{usage.toolCalls}</Pair>
          <Pair term="Tokens">
            {formatCount(usage.inputTokens)} in, {formatCount(usage.outputTokens)} out
          </Pair>
          <Pair term="Estimated cost">{formatCost(usage.costUsd)}</Pair>
        </dl>
      </Section>
      <Section title="Tickets">
        {state.tickets.length > 0 ? (
          state.tickets.map((ticket) => (
            <div key={ticket.id} className="flex flex-col gap-12 rounded-cards bg-onyx-canvas p-16">
              <p className="text-body-sm font-w480">{ticket.id}</p>
              <div className="flex flex-wrap gap-8">
                <Tag>{TICKET_REASONS[ticket.reason]}</Tag>
                <Tag>{ticket.priority} priority</Tag>
              </div>
              <p className="text-caption leading-[1.45] text-ash-text">{ticket.summary}</p>
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
];

function FailuresPanel({ turnCount, onInject }: { turnCount: number; onInject: Props["onInject"] }) {
  // Which failures were queued, and at which point of the conversation. A new turn uses them up.
  const [queued, setQueued] = useState<Record<string, number>>({});
  return (
    <>
      <Quiet>Queue a simulated failure for your next message. It runs through the same code path as a real one.</Quiet>
      <ul className="flex flex-col gap-12">
        {FAILURES.map((failure) => (
          <li key={failure.kind} className="flex items-center gap-16 rounded-cards bg-onyx-canvas p-16">
            <div className="flex min-w-0 flex-1 flex-col gap-4">
              <p className="text-body-sm font-w480 leading-[1.45]">{failure.title}</p>
              <p className="text-caption leading-[1.45] text-ash-text">{failure.effect}</p>
            </div>
            {queued[failure.kind] === turnCount ? (
              <Tag>Queued</Tag>
            ) : (
              <Button
                variant="surface"
                onClick={() => {
                  onInject(failure.kind, failure.tool);
                  setQueued((before) => ({ ...before, [failure.kind]: turnCount }));
                }}
              >
                Queue
              </Button>
            )}
          </li>
        ))}
      </ul>
    </>
  );
}

interface Props {
  state: AppState;
  turnId: string | null; // null follows the newest turn
  onSelectTurn: (turnId: string | null) => void;
  onClose: () => void;
  onInject: (kind: FaultInject["kind"], tool?: string) => void;
}

export function Drawer({ state, turnId, onSelectTurn, onClose, onInject }: Props) {
  const [tab, setTab] = useState<Tab>("turn");
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

  return (
    <aside
      id="observability"
      aria-label="Observability"
      className="flex w-480 shrink-0 flex-col overflow-hidden rounded-cards bg-graphite-card max-lg:fixed max-lg:inset-12 max-lg:z-20 max-lg:w-auto lg:mb-16 lg:mr-16"
    >
      <div className="flex flex-col gap-20 p-32 pb-20">
        <div className="flex items-center justify-between gap-12">
          <h2 className="font-arcadiadisplay text-subheading font-w480 tracking-[0.02em]">Observability</h2>
          <Button variant="quiet" onClick={onClose} aria-label="Close observability" className="w-40 px-0">
            <Glyph name="cross" />
          </Button>
        </div>
        <div role="tablist" className="flex gap-4 rounded-nav bg-onyx-canvas p-4">
          {tabs.map((t) => (
            <button
              key={t.id}
              type="button"
              role="tab"
              aria-selected={shown === t.id}
              onClick={() => setTab(t.id)}
              className={`h-32 flex-1 rounded-nav px-12 text-caption font-w480 ${
                shown === t.id ? "bg-obsidian-button text-ivory-text" : "text-ash-text hover:text-ivory-text"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
        {perTurn && turn ? (
          <div className="flex items-center justify-between gap-12">
            <p className="min-w-0 truncate text-body-sm leading-[1.45] text-ash-text">
              Turn {index + 1} of {turns.length}: “{turn.userText}”
            </p>
            <div className="flex shrink-0 gap-4">
              <Button variant="quiet" disabled={index <= 0} onClick={() => go(index - 1)} aria-label="Earlier turn" className="h-32 w-32 px-0">
                <Glyph name="left" />
              </Button>
              <Button
                variant="quiet"
                disabled={index >= turns.length - 1}
                onClick={() => go(index + 1)}
                aria-label="Later turn"
                className="h-32 w-32 px-0"
              >
                <Glyph name="right" />
              </Button>
            </div>
          </div>
        ) : null}
      </div>

      <div className="flex min-h-0 flex-1 flex-col gap-32 overflow-y-auto px-32 pb-32">
        {shown === "session" ? <SessionPanel state={state} /> : null}
        {shown === "failures" ? <FailuresPanel turnCount={turns.length} onInject={onInject} /> : null}
        {perTurn && !turn ? <Quiet>Send a message to see what the assistant does with it.</Quiet> : null}
        {shown === "turn" && turn ? <TurnPanel turn={turn} status={shownStatus(state, turn)} /> : null}
        {shown === "prompts" && turn ? <PromptsPanel turn={turn} /> : null}
      </div>
    </aside>
  );
}
