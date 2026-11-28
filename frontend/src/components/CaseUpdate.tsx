import { CASE_LABELS } from "@/lib/labels";
import type { CaseUpdated } from "@/lib/protocol";

import { Glyph } from "./ui";

/**
 * Something a person at the service desk did to one of this conversation's tickets.
 *
 * Words written for the customer, a reply or an approved refund, are the one inverted card on
 * the page: a person is talking, not the assistant. A change of hands or of status is one line.
 */
export function CaseUpdate({ update }: { update: CaseUpdated }) {
  if (update.kind === "reply" || update.kind === "refund") {
    return (
      <article className="flex flex-col gap-12 rounded-md bg-obsidian p-16 text-pure-white" aria-label={`Reply from ${update.author}`}>
        <p className="label-mono flex flex-wrap items-center gap-8 text-ash">
          <Glyph name={update.kind === "refund" ? "card" : "reply"} />
          {update.kind === "refund" ? "Refund approved by the support team" : "Reply from the support team"}
          <span className="text-smoke">{update.ticket_id}</span>
        </p>
        <p className="whitespace-pre-wrap break-words text-body">{update.body}</p>
        <p className="font-mono text-label text-smoke">
          {update.author} · {CASE_LABELS[update.status]}
        </p>
      </article>
    );
  }
  const what =
    update.kind === "assigned"
      ? update.assignee
        ? `${update.assignee} picked this up`
        : "back in the queue, waiting for someone to pick it up"
      : update.status === "resolved"
        ? `marked resolved by ${update.author}`
        : update.status === "pending"
          ? `${update.author} is waiting for your answer`
          : `reopened by ${update.author}`;
  return (
    <p className="flex items-center gap-8 font-mono text-label text-stone" role="status">
      <Glyph name={update.status === "resolved" ? "check" : "person"} className={update.status === "resolved" ? "text-terminal-green" : ""} />
      <span>
        <span className="text-obsidian">{update.ticket_id}</span> · {what}
      </span>
    </p>
  );
}
