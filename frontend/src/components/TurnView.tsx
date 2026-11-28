import { type ReactNode, useEffect, useState } from "react";

import { BAND_LABELS, CASE_LABELS, FAILURE_LABELS, PLAN_LABELS, TICKET_REASONS, TOOL_LABELS, formatCost, formatCount, formatMs, shortModel } from "@/lib/labels";
import type { CaseUpdated, RetrievalHit, Ticket } from "@/lib/protocol";
import { type Passage, type Step, type ToolStep, type Turn, type TurnStatus, hasTrace, heardOf, totalCost, turnUsage } from "@/lib/state";

import { type Cites, RichText, citedNumbers } from "./RichText";
import { Button, Caption, Card, Glyph, type GlyphName, Spinner, Tag } from "./ui";

/** One line of what the agent is doing or has done. */
function Row({ icon, title, detail, meta, children }: { icon: ReactNode; title: string; detail?: string | null; meta?: string; children?: ReactNode }) {
  return (
    <li className="flex items-start gap-8 font-mono text-caption">
      <span className="flex h-20 w-16 shrink-0 items-center">{icon}</span>
      <div className="flex min-w-0 flex-1 flex-col gap-2">
        <p className="text-obsidian">
          {title}
          {detail ? <span className="text-stone"> · {detail}</span> : null}
        </p>
        {children}
      </div>
      {meta ? <span className="ml-8 shrink-0 tabular-nums text-smoke">{meta}</span> : null}
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
    return <Row icon={<Glyph name="stop" className="text-stone" />} title={label.done} detail="did not finish" />;
  }
  const band = step.retrieval ? BAND_LABELS[step.retrieval.band] : null;
  return (
    <Row
      icon={<Glyph name={OUTCOME_GLYPH[step.status]} className={step.status === "ok" ? "text-terminal-green" : "text-stone"} />}
      title={label.done}
      detail={step.summary}
      meta={formatMs(step.durationMs)}
    >
      {band ? <p className="text-label text-smoke">{band}</p> : null}
      {step.attempt > 1 ? (
        <p className="text-label text-smoke">
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
        icon={<Glyph name="clock" className="text-stone" />}
        title={`The model ${FAILURE_LABELS[step.reason]}. Trying again${wait}`}
        detail={`attempt ${step.attempt} of ${step.maxAttempts}`}
      />
    );
  }
  return (
    <Row
      icon={<Glyph name="swap" className="text-stone" />}
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
    return <Row icon={<Glyph name="minus" className="text-stone" />} title="The grounding check did not run" />;
  }
  // The statements themselves stay in the observability drawer. Here they would put in front of
  // the customer the very claims the check refused to let through.
  const count = verdict.unsupported.length;
  const what = count === 1 ? "a statement" : count > 1 ? `${count} statements` : "statements";
  return <Row icon={<Glyph name="cross" />} title={`The check found ${what} the help center does not support`} meta={formatMs(verdict.ms)} />;
}

/**
 * The ticket as the customer holds it. `latest` is the last thing the service desk told them
 * about it, and with it the card says where the ticket stands now and who has it.
 */
function TicketCard({ ticket, latest, onDesk }: { ticket: Ticket; latest: CaseUpdated | null; onDesk: () => void }) {
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
    <Card className="flex flex-col gap-12 p-16">
      <div className="flex flex-wrap items-center gap-8 text-stone">
        <Glyph name="person" />
        <Caption>Handed to a person</Caption>
        <Tag tone={latest?.status === "resolved" ? "good" : "plain"} className="ml-auto">
          {latest?.status === "resolved" ? <Glyph name="check" className="mr-4" /> : null}
          {CASE_LABELS[latest?.status ?? "open"]}
        </Tag>
      </div>
      <p className="font-mono text-body font-medium text-obsidian">{ticket.id}</p>
      <div className="flex flex-wrap gap-8">
        <Tag>{TICKET_REASONS[ticket.reason]}</Tag>
        <Tag>{ticket.priority} priority</Tag>
      </div>
      <p className="whitespace-pre-wrap break-words text-body-sm text-charcoal">{ticket.summary}</p>
      <p className="font-mono text-label text-stone">
        {latest?.assignee ? `${latest.assignee} has this ticket.` : "Waiting for someone at the service desk to pick it up."} Replies show up
        in this conversation.
      </p>
      <div className="flex flex-wrap gap-8">
        <Button variant="ghost" size="sm" onClick={onDesk}>
          <Glyph name="inbox" /> Open in the service desk
        </Button>
        <Button variant="quiet" size="sm" onClick={copy}>
          {copied ? "Copied" : "Copy case summary"}
        </Button>
      </div>
    </Card>
  );
}

/** What an action did, laid out from the tool's own record and not from the assistant's wording. */
function Receipt({ receipt }: { receipt: Record<string, string> }) {
  if (receipt.kind !== "refund") return null;
  return (
    <Card className="flex flex-col gap-12 p-16">
      <div className="flex items-center gap-8 text-stone">
        <Glyph name="check" className="text-terminal-green" />
        <Caption>Refund issued</Caption>
      </div>
      <p className="text-heading tabular-nums text-obsidian">{receipt.amount}</p>
      <dl className="grid grid-cols-[auto_1fr] items-baseline gap-x-16 gap-y-4 text-body-sm">
        <dt className="label-mono text-stone">Invoice</dt>
        <dd className="font-mono tabular-nums text-obsidian">{receipt.invoice_id}</dd>
        <dt className="label-mono text-stone">Refund</dt>
        <dd className="font-mono tabular-nums text-obsidian">{receipt.refund_id}</dd>
        <dt className="label-mono text-stone">Date</dt>
        <dd className="font-mono tabular-nums text-obsidian">{receipt.date}</dd>
      </dl>
      <p className="text-caption text-stone">From the refund itself, as the billing tool recorded it.</p>
    </Card>
  );
}

function SourceCard({ id, hit, passage, onClose }: { id: string; hit: RetrievalHit; passage: Passage | undefined; onClose: () => void }) {
  return (
    <Card id={id} className="flex flex-col gap-12 p-16">
      <div className="flex items-start justify-between gap-12">
        <div className="flex min-w-0 flex-col gap-4">
          <Caption>Source {hit.n}</Caption>
          <p className="text-body font-medium text-obsidian">{hit.source}</p>
        </div>
        <Button variant="quiet" size="sm" onClick={onClose} aria-label="Close source" className="w-28 px-0">
          <Glyph name="cross" />
        </Button>
      </div>
      {passage === undefined || passage === "loading" ? (
        <p className="flex items-center gap-8 text-body-sm text-stone">
          <Spinner /> Loading the passage…
        </p>
      ) : passage === "missing" ? (
        <p className="text-body-sm text-stone">This passage is no longer available.</p>
      ) : (
        <div className="max-h-320 overflow-y-auto">
          <RichText text={passage.content} small />
        </div>
      )}
      <p className="text-caption text-stone">
        The help center passage the assistant was given. Written for {hit.tiers.map((tier) => PLAN_LABELS[tier]).join(", ")}.
      </p>
    </Card>
  );
}

function Stopped({ title, body, action, onAction }: { title: string; body: string; action: string | null; onAction: () => void }) {
  return (
    <Card className="flex flex-col items-start gap-12 p-16">
      <div className="flex items-start gap-8">
        <Glyph name="alert" className="mt-4" />
        <div className="flex flex-col gap-4">
          <p className="text-body font-medium text-obsidian">{title}</p>
          <p className="text-body-sm text-charcoal">{body}</p>
        </div>
      </div>
      {action ? (
        <Button variant={action === "Try again" ? "primary" : "ghost"} onClick={onAction}>
          {action}
        </Button>
      ) : null}
    </Card>
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
  voiceOn: boolean; // voice is on in the page
  updates: CaseUpdated[]; // what the service desk has told this conversation, oldest first
  onDesk: (ticketId: string) => void; // show a ticket at the service desk
}

export function TurnView({ turn, status, canRetry, onRetry, onInspect, passages, onPassage, voiceOn, updates, onDesk }: Props) {
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
  // A spoken turn is heard while voice is on. Reading it out as well would talk over the voice.
  const live = turn.mode === "voice" && voiceOn ? "off" : "polite";

  const toggleSource = (n: number) => {
    const hit = cited.find((candidate) => candidate.n === n);
    if (!hit) return;
    setOpenSource((open) => (open === n ? null : n));
    if (typeof passages[hit.chunk_id] !== "object") onPassage(hit.chunk_id);
  };
  // The page drops the passages it holds when another customer is verified: who may read one is
  // the server's decision for whoever is verified now. A card that is open asks again.
  const dropped = shown && passages[shown.chunk_id] === undefined ? shown.chunk_id : null;
  useEffect(() => {
    if (dropped) onPassage(dropped);
  }, [dropped, onPassage]);
  const cites: Cites = { valid: new Set(cited.map((hit) => hit.n)), open: openSource, onCite: toggleSource };
  const ended = status !== "running" && hasTrace(turn);
  // A spoken reply the visitor cut off: what they heard of it, which is all the assistant will
  // assume they know. Null when the reply was not cut off.
  const heard = heardOf(turn);
  // A spoken reply that was stopped says what was heard in its own notice below, once.
  const stoppedReply = status === "interrupted" && turn.mode === "voice" && turn.resumable === false;
  const cutShort = heard !== null && turn.answer !== null && !stoppedReply;

  return (
    <article className="flex flex-col gap-20">
      <div className="flex flex-col items-end gap-4">
        <p className="max-w-[85%] whitespace-pre-wrap break-words rounded-md bg-hairline px-16 py-8 text-obsidian">{turn.userText}</p>
        {turn.mode === "voice" ? (
          <p className="label-mono flex items-center gap-4 text-stone">
            <Glyph name="mic" /> Spoken
          </p>
        ) : null}
      </div>

      <div className="flex flex-col gap-16">
        {/* Each step is a short line, so a screen reader may read them as they come. */}
        <div aria-live={live}>
          {turn.steps.length > 0 || thinking || checking || turn.verdict ? (
            <ol className="flex flex-col gap-8 rounded-md bg-pure-white px-16 py-12 shadow-card">
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
          <p className="flex items-start gap-8 text-body-sm text-charcoal">
            <Glyph name="alert" className="mt-2" />
            The first answer was withdrawn. It could not be confirmed against the help center.
          </p>
        ) : null}

        {cutShort ? (
          <p className="flex items-start gap-8 text-body-sm text-charcoal">
            <Glyph name="stop" className="mt-2" />
            {heard ? `This reply was cut off. You heard: “${heard}”` : "This reply was cut off before any of it was said."}
          </p>
        ) : null}

        {/* The answer arrives token by token. Busy until it is whole, so it is read out once. */}
        <div aria-live={live} aria-busy={running && !turn.answer} className={cutShort ? "opacity-60" : undefined}>
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
                aria-controls={shown?.n === hit.n ? sourceId : undefined}
                onClick={() => toggleSource(hit.n)}
                className={`inline-flex min-h-28 items-center rounded-md px-8 py-4 text-left text-caption shadow-hairline ${
                  openSource === hit.n ? "bg-obsidian text-pure-white" : "bg-pure-white text-charcoal hover:bg-paper-white hover:text-obsidian"
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
          <TicketCard
            key={ticket.id}
            ticket={ticket}
            latest={updates.findLast((update) => update.ticket_id === ticket.id) ?? null}
            onDesk={() => onDesk(ticket.id)}
          />
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
          turn.mode === "voice" && turn.resumable === false ? (
            <Stopped
              title="Reply stopped"
              body={
                heard
                  ? `You heard: “${heard}”. The rest was not said.`
                  : "It was stopped before anything was said. What you say next carries on from here."
              }
              action={null}
              onAction={onRetry}
            />
          ) : turn.resumable === false ? (
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
          <button type="button" onClick={onInspect} className="-my-8 self-start rounded-md py-8 font-mono text-label text-stone hover:text-obsidian">
            {formatMs(turn.totalMs)} · {formatCount(usage.inputTokens + usage.outputTokens)} tokens · {formatCost(totalCost(usage))} ·{" "}
            <span className="underline decoration-ash underline-offset-2">View trace</span>
          </button>
        ) : null}
      </div>
    </article>
  );
}
