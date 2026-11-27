import { type ReactNode, useState } from "react";

import { BAND_LABELS, FAILURE_LABELS, PLAN_LABELS, TICKET_REASONS, TOOL_LABELS, formatCost, formatCount, formatMs, shortModel } from "@/lib/labels";
import type { RetrievalHit, Ticket } from "@/lib/protocol";
import { type Passage, type Step, type ToolStep, type Turn, type TurnStatus, turnUsage } from "@/lib/state";

import { type Cites, RichText, citedNumbers } from "./RichText";
import { Button, Caption, Glyph, type GlyphName, Spinner, Tag } from "./ui";

/** One line of what the agent is doing or has done. */
function Row({ icon, title, detail, meta, children }: { icon: ReactNode; title: string; detail?: string | null; meta?: string; children?: ReactNode }) {
  return (
    <li className="flex items-start gap-12">
      <span className="mt-2">{icon}</span>
      <div className="flex min-w-0 flex-1 flex-col gap-4">
        <p className="text-body-sm leading-[1.45]">
          {title}
          {detail ? <span className="text-ash-text"> · {detail}</span> : null}
        </p>
        {children}
      </div>
      {meta ? <span className="mt-4 shrink-0 text-caption tabular-nums text-ash-text">{meta}</span> : null}
    </li>
  );
}

const OUTCOME_GLYPH: Record<"ok" | "refused" | "failed" | "stopped", GlyphName> = { ok: "check", refused: "minus", failed: "cross", stopped: "stop" };

function ToolRow({ step }: { step: ToolStep }) {
  const label = TOOL_LABELS[step.name] ?? { running: step.name, done: step.name };
  if (step.status === "running") {
    return <Row icon={<Spinner />} title={`${label.running}…`} />;
  }
  if (step.status === "retrying") {
    const detail = `attempt ${step.attempt} of ${step.maxAttempts ?? "?"}, the last one failed`;
    return <Row icon={<Spinner />} title={`${label.running}…`} detail={detail} />;
  }
  if (step.status === "stopped") {
    return <Row icon={<Glyph name="stop" className="text-ash-text" />} title={label.done} detail="did not finish" />;
  }
  const band = step.retrieval ? BAND_LABELS[step.retrieval.band] : null;
  return (
    <Row
      icon={<Glyph name={OUTCOME_GLYPH[step.status]} className={step.status === "ok" ? "" : "text-ash-text"} />}
      title={label.done}
      detail={step.summary}
      meta={formatMs(step.durationMs)}
    >
      {band ? <p className="text-caption leading-[1.4] text-ash-text">{band}</p> : null}
      {step.attempt > 1 ? (
        <p className="text-caption leading-[1.4] text-ash-text">
          {step.status === "ok" ? `Worked on attempt ${step.attempt}` : `Gave up after ${step.attempt} attempts`}
          {step.error ? `. An earlier attempt failed: ${step.error}` : ""}
        </p>
      ) : null}
    </Row>
  );
}

function StepRow({ step }: { step: Step }) {
  if (step.kind === "tool") return <ToolRow step={step} />;
  if (step.kind === "retry") {
    const wait = step.waitMs >= 100 ? ` in ${formatMs(step.waitMs)}` : "";
    return (
      <Row
        icon={<Glyph name="clock" />}
        title={`The model ${FAILURE_LABELS[step.reason]}. Trying again${wait}`}
        detail={`attempt ${step.attempt} of ${step.maxAttempts}`}
      />
    );
  }
  return (
    <Row
      icon={<Glyph name="swap" />}
      title={`Switched to the backup model ${shortModel(step.toModel)}`}
      detail={`${shortModel(step.fromModel)} ${FAILURE_LABELS[step.reason]}`}
    />
  );
}

function VerdictRow({ turn }: { turn: Turn }) {
  const verdict = turn.verdict;
  if (!verdict) return null;
  if (verdict.verdict === "grounded") {
    const how = verdict.held ? "before it was shown" : undefined;
    return <Row icon={<Glyph name="shield" />} title="Answer checked against the help center" detail={how} meta={formatMs(verdict.ms)} />;
  }
  if (verdict.verdict === "skipped") {
    return <Row icon={<Glyph name="minus" className="text-ash-text" />} title="The grounding check did not run" />;
  }
  // The statements themselves stay in the observability drawer. Here they would put in front of
  // the customer the very claims the check refused to let through.
  const count = verdict.unsupported.length;
  const what = count === 1 ? "a statement" : count > 1 ? `${count} statements` : "statements";
  return <Row icon={<Glyph name="cross" />} title={`The check found ${what} the help center does not support`} meta={formatMs(verdict.ms)} />;
}

function TicketCard({ ticket }: { ticket: Ticket }) {
  const [copied, setCopied] = useState(false);
  const summary = `Ticket ${ticket.id} (${TICKET_REASONS[ticket.reason]}, ${ticket.priority} priority)\n${ticket.summary}`;
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(summary);
      setCopied(true);
    } catch {
      setCopied(false); // the browser said no. The summary is on screen to select by hand.
    }
  };
  return (
    <div className="flex flex-col gap-12 rounded-cards bg-graphite-card p-24">
      <div className="flex items-center gap-8 text-ash-text">
        <Glyph name="person" />
        <Caption>Handed to a person</Caption>
      </div>
      <p className="font-arcadiadisplay text-subheading font-w480">{ticket.id}</p>
      <div className="flex flex-wrap gap-8">
        <Tag>{TICKET_REASONS[ticket.reason]}</Tag>
        <Tag>{ticket.priority} priority</Tag>
      </div>
      <p className="whitespace-pre-wrap break-words text-body-sm leading-[1.45] text-ash-text">{ticket.summary}</p>
      <Button variant="surface" onClick={copy} className="self-start">
        {copied ? "Copied" : "Copy case summary"}
      </Button>
    </div>
  );
}

/** What an action did, laid out from the tool's own record and not from the assistant's wording. */
function Receipt({ receipt }: { receipt: Record<string, string> }) {
  if (receipt.kind !== "refund") return null;
  return (
    <div className="flex flex-col gap-12 rounded-cards bg-graphite-card p-24">
      <div className="flex items-center gap-8 text-ash-text">
        <Glyph name="check" />
        <Caption>Refund issued</Caption>
      </div>
      <p className="font-arcadiadisplay text-subheading font-w480 tabular-nums">{receipt.amount}</p>
      <dl className="grid grid-cols-[auto_1fr] gap-x-16 gap-y-8 text-body-sm leading-[1.45]">
        <dt className="text-ash-text">Invoice</dt>
        <dd className="font-mono tabular-nums">{receipt.invoice_id}</dd>
        <dt className="text-ash-text">Refund</dt>
        <dd className="font-mono tabular-nums">{receipt.refund_id}</dd>
        <dt className="text-ash-text">Date</dt>
        <dd className="tabular-nums">{receipt.date}</dd>
      </dl>
      <p className="text-caption leading-[1.4] text-ash-text">From the refund itself, as the billing tool recorded it.</p>
    </div>
  );
}

function SourceCard({ id, hit, passage, onClose }: { id: string; hit: RetrievalHit; passage: Passage | undefined; onClose: () => void }) {
  return (
    <div id={id} className="flex flex-col gap-12 rounded-cards bg-graphite-card p-24">
      <div className="flex items-start justify-between gap-12">
        <div className="flex min-w-0 flex-col gap-4">
          <Caption>Source {hit.n}</Caption>
          <p className="font-w480 leading-[1.35]">{hit.source}</p>
        </div>
        <Button variant="quiet" onClick={onClose} aria-label="Close source" className="w-40 px-0">
          <Glyph name="cross" />
        </Button>
      </div>
      {passage === undefined || passage === "loading" ? (
        <p className="flex items-center gap-12 text-body-sm text-ash-text">
          <Spinner /> Loading the passage…
        </p>
      ) : passage === "missing" ? (
        <p className="text-body-sm leading-[1.45] text-ash-text">This passage is no longer available.</p>
      ) : (
        <p className="max-h-320 overflow-y-auto whitespace-pre-wrap break-words text-body-sm leading-[1.5]">{passage.content}</p>
      )}
      <p className="text-caption leading-[1.4] text-ash-text">
        The help center passage the assistant was given. Written for {hit.tiers.map((tier) => PLAN_LABELS[tier]).join(", ")}.
      </p>
    </div>
  );
}

function Stopped({ title, body, action, onAction }: { title: string; body: string; action: string | null; onAction: () => void }) {
  return (
    <div className="flex flex-col items-start gap-16 rounded-cards bg-graphite-card p-24">
      <div className="flex items-start gap-12">
        <Glyph name="alert" className="mt-4" />
        <div className="flex flex-col gap-4">
          <p className="font-w480">{title}</p>
          <p className="text-body-sm leading-[1.45] text-ash-text">{body}</p>
        </div>
      </div>
      {action ? <Button onClick={onAction}>{action}</Button> : null}
    </div>
  );
}

interface Props {
  turn: Turn;
  status: TurnStatus; // what to show, which can differ from what the stream last said
  canRetry: boolean;
  onRetry: () => void;
  onInspect: () => void;
  passages: Record<string, Passage>;
  onPassage: (chunkId: string) => void; // ask the server for a passage's text
}

export function TurnView({ turn, status, canRetry, onRetry, onInspect, passages, onPassage }: Props) {
  const [openSource, setOpenSource] = useState<number | null>(null);
  const running = status === "running";
  const text = turn.answer?.text ?? turn.streaming;
  const toolBusy = turn.steps.some((s) => s.kind === "tool" && (s.status === "running" || s.status === "retrying"));
  const thinking = running && !toolBusy && !text && turn.phase !== "checking";
  const checking = running && turn.phase === "checking";
  const hits = turn.answer?.hits ?? [];
  const cited = turn.answer ? citedNumbers(turn.answer.text).flatMap((n) => hits.filter((hit) => hit.n === n)) : [];
  const shown = cited.find((hit) => hit.n === openSource) ?? null;
  const usage = turnUsage(turn);
  const receipts = turn.steps.flatMap((step) => (step.kind === "tool" && step.receipt ? [step.receipt] : []));
  const sourceId = `source-${turn.id}`;

  const toggleSource = (n: number) => {
    const hit = cited.find((candidate) => candidate.n === n);
    if (!hit) return;
    setOpenSource((open) => (open === n ? null : n));
    if (typeof passages[hit.chunk_id] !== "object") onPassage(hit.chunk_id);
  };
  const cites: Cites = { valid: new Set(cited.map((hit) => hit.n)), open: openSource, onCite: toggleSource };
  const ended = status !== "running" && (turn.calls.length > 0 || turn.steps.length > 0);

  return (
    <article className="flex flex-col gap-20">
      <div className="flex justify-end">
        <p className="max-w-[85%] whitespace-pre-wrap break-words rounded-cards bg-obsidian-button px-20 py-12">{turn.userText}</p>
      </div>

      <div className="flex flex-col gap-16">
        {/* Each step is a short line, so a screen reader may read them as they come. */}
        <div aria-live="polite">
          {turn.steps.length > 0 || thinking || checking || turn.verdict ? (
            <ol className="flex flex-col gap-12 rounded-cards bg-graphite-card px-24 py-20">
              {turn.steps.map((step) => (
                <StepRow key={step.id} step={step} />
              ))}
              {thinking ? <Row icon={<Spinner />} title="Thinking…" /> : null}
              {checking ? <Row icon={<Spinner />} title="Checking the answer against the help center…" /> : null}
              <VerdictRow turn={turn} />
            </ol>
          ) : null}
        </div>

        {turn.retracted ? (
          <p className="flex items-start gap-12 text-body-sm leading-[1.45] text-ash-text">
            <Glyph name="alert" className="mt-2" />
            The first answer was withdrawn. It could not be confirmed against the help center.
          </p>
        ) : null}

        {/* The answer arrives token by token. Busy until it is whole, so it is read out once. */}
        <div aria-live="polite" aria-busy={running && !turn.answer}>
          {text ? <RichText text={text} caret={running && !turn.answer} cites={turn.answer ? cites : undefined} /> : null}
        </div>

        {cited.length > 0 ? (
          <div className="flex flex-wrap items-center gap-8">
            <Caption>Sources</Caption>
            {cited.map((hit) => (
              <button
                key={hit.n}
                type="button"
                aria-expanded={openSource === hit.n}
                aria-controls={sourceId}
                onClick={() => toggleSource(hit.n)}
                className={`inline-flex min-h-32 items-center rounded-tags px-12 py-4 text-left text-caption font-w480 leading-[1.3] hover:text-ivory-text ${
                  openSource === hit.n ? "bg-ivory-text text-onyx-canvas hover:text-onyx-canvas" : "bg-obsidian-button text-ash-text"
                }`}
              >
                {hit.n} · {hit.source}
              </button>
            ))}
          </div>
        ) : null}
        {shown ? <SourceCard id={sourceId} hit={shown} passage={passages[shown.chunk_id]} onClose={() => setOpenSource(null)} /> : null}

        {receipts.map((receipt) => (
          <Receipt key={receipt.refund_id ?? receipt.kind} receipt={receipt} />
        ))}

        {turn.tickets.map((ticket) => (
          <TicketCard key={ticket.id} ticket={ticket} />
        ))}

        {status === "failed" ? (
          <Stopped
            title="The assistant could not finish this reply"
            body={
              turn.failure?.code === "internal"
                ? "Something went wrong on our side. Your conversation is saved."
                : turn.verdict?.verdict === "skipped"
                  ? "The answer could not be checked against the help center, so it was held back. Trying again runs the check."
                  : `No model answered (${turn.failure?.message ?? "unknown"}). Your conversation is saved, and trying again picks up where it stopped.`
            }
            action={canRetry ? "Try again" : null}
            onAction={onRetry}
          />
        ) : null}
        {status === "interrupted" ? (
          turn.resumable === false ? (
            <Stopped
              title="Stopped before it began"
              body="Nothing of this message was saved. Send it again if you still need an answer."
              action={null}
              onAction={onRetry}
            />
          ) : (
            <Stopped
              title="Stopped before the reply was finished"
              body="Nothing was lost. Steps that had finished stay done."
              action={canRetry ? "Resume" : null}
              onAction={onRetry}
            />
          )
        ) : null}

        {ended ? (
          <button type="button" onClick={onInspect} className="-my-8 self-start rounded-md py-8 text-caption leading-[1.4] text-ash-text hover:text-ivory-text">
            {formatMs(turn.totalMs)} · {formatCount(usage.inputTokens + usage.outputTokens)} tokens · {formatCost(usage.costUsd)} · View trace
          </button>
        ) : null}
      </div>
    </article>
  );
}
