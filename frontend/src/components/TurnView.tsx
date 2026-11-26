import type { ReactNode } from "react";

import { BAND_LABELS, FAILURE_LABELS, TICKET_REASONS, TOOL_LABELS, formatCost, formatCount, formatMs, shortModel } from "@/lib/labels";
import type { Ticket } from "@/lib/protocol";
import { type Step, type ToolStep, type Turn, type TurnStatus, turnUsage } from "@/lib/state";

import { RichText, citedNumbers } from "./RichText";
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

const OUTCOME_GLYPH: Record<"ok" | "refused" | "failed", GlyphName> = { ok: "check", refused: "minus", failed: "cross" };

function ToolRow({ step }: { step: ToolStep }) {
  const label = TOOL_LABELS[step.name] ?? { running: step.name, done: step.name };
  if (step.status === "running") {
    return <Row icon={<Spinner />} title={`${label.running}…`} />;
  }
  if (step.status === "retrying") {
    const detail = `attempt ${step.attempt} of ${step.maxAttempts ?? "?"}, the last one failed`;
    return <Row icon={<Spinner />} title={`${label.running}…`} detail={detail} />;
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
  return (
    <Row icon={<Glyph name="cross" />} title="The check found statements the help center does not support" meta={formatMs(verdict.ms)}>
      <ul className="flex flex-col gap-4 text-caption leading-[1.4] text-ash-text">
        {verdict.unsupported.map((claim, i) => (
          <li key={i}>“{claim}”</li>
        ))}
      </ul>
    </Row>
  );
}

function TicketCard({ ticket }: { ticket: Ticket }) {
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
}

export function TurnView({ turn, status, canRetry, onRetry, onInspect }: Props) {
  const running = status === "running";
  const text = turn.answer?.text ?? turn.streaming;
  const toolBusy = turn.steps.some((s) => s.kind === "tool" && (s.status === "running" || s.status === "retrying"));
  const thinking = running && !toolBusy && !text && turn.phase !== "checking";
  const checking = running && turn.phase === "checking";
  const cited = turn.answer ? citedNumbers(turn.answer.text).flatMap((n) => turn.answer?.hits.filter((hit) => hit.n === n) ?? []) : [];
  const usage = turnUsage(turn);

  return (
    <article className="flex flex-col gap-20">
      <div className="flex justify-end">
        <p className="max-w-[85%] whitespace-pre-wrap break-words rounded-cards bg-obsidian-button px-20 py-12">{turn.userText}</p>
      </div>

      <div className="flex flex-col gap-16" aria-live="polite">
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

        {turn.retracted ? (
          <p className="flex items-start gap-12 text-body-sm leading-[1.45] text-ash-text">
            <Glyph name="alert" className="mt-2" />
            The first answer was withdrawn. It could not be confirmed against the help center.
          </p>
        ) : null}

        {text ? <RichText text={text} caret={running && !turn.answer} /> : null}

        {cited.length > 0 ? (
          <div className="flex flex-wrap items-center gap-8">
            <Caption>Sources</Caption>
            {cited.map((hit) => (
              <Tag key={hit.n}>
                {hit.n} · {hit.source}
              </Tag>
            ))}
          </div>
        ) : null}

        {turn.tickets.map((ticket) => (
          <TicketCard key={ticket.id} ticket={ticket} />
        ))}

        {status === "failed" ? (
          <Stopped
            title="The assistant could not finish this reply"
            body={
              turn.failure?.code === "internal"
                ? "Something went wrong on our side. Your conversation is saved."
                : `No model answered (${turn.failure?.message ?? "unknown"}). Your conversation is saved, and trying again picks up where it stopped.`
            }
            action={canRetry ? "Try again" : null}
            onAction={onRetry}
          />
        ) : null}
        {status === "interrupted" ? (
          <Stopped
            title="Stopped before the reply was finished"
            body="Nothing was lost. Steps that had finished stay done."
            action={canRetry ? "Resume" : null}
            onAction={onRetry}
          />
        ) : null}

        {turn.finished ? (
          <button type="button" onClick={onInspect} className="self-start rounded-md text-caption leading-[1.4] text-ash-text hover:text-ivory-text">
            {formatMs(turn.totalMs)} · {formatCount(usage.inputTokens + usage.outputTokens)} tokens · {formatCost(usage.costUsd)} · View trace
          </button>
        ) : null}
      </div>
    </article>
  );
}
