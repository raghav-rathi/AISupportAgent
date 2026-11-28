"use client";

import { type ReactNode, useState } from "react";

import type { Desk } from "@/lib/desk";
import { BAND_LABELS, PLAN_LABELS, STATUS_LABELS, TICKET_REASONS, TOOL_LABELS, formatAgo, formatUsd, formatWhen } from "@/lib/labels";
import type { AccountInvoice, AccountView, Activity, DeskDetail, DeskTicket, Line, Receipt } from "@/lib/protocol";

import { Button, Caption, Card, Glyph, type GlyphName, Spinner, Tag } from "./ui";

const STATUSES: DeskTicket["status"][] = ["open", "pending", "resolved"];
const PRIORITIES: DeskTicket["priority"][] = ["low", "normal", "high", "urgent"];

const CHECKS: Record<NonNullable<Receipt["check"]>, string> = {
  grounded: "Every statement in the reply is backed by a tool result or a help center passage",
  ungrounded: "The reply said something no source supports, so it was taken back",
  skipped: "The reply was not checked",
};

const OUTCOME: Record<Receipt["steps"][number]["outcome"], { glyph: GlyphName; tone: string }> = {
  ok: { glyph: "check", tone: "text-terminal-green" },
  refused: { glyph: "minus", tone: "text-stone" },
  failed: { glyph: "cross", tone: "text-obsidian" },
};

/** One labelled line of the receipt: the stamp on the left, what it says on the right. */
function Entry({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="grid grid-cols-[96px_minmax(0,1fr)] gap-12 border-t border-hairline px-16 py-12 first:border-t-0 max-sm:grid-cols-1 max-sm:gap-4">
      <dt className="label-mono pt-2 text-stone">{label}</dt>
      <dd className="min-w-0 text-body-sm text-obsidian">{children}</dd>
    </div>
  );
}

/**
 * What the assistant did before it stopped. The server reads this from the stored events of the
 * turn, so it is a record and not a model's account of itself.
 */
function ReceiptCard({ receipt }: { receipt: Receipt }) {
  return (
    <Card>
      <div className="flex flex-wrap items-center gap-8 px-16 py-12">
        <Caption>Hand-off receipt</Caption>
        <span className="font-mono text-label text-smoke">from the stored events of the turn</span>
        {receipt.channel === "voice" ? (
          <Tag className="ml-auto">
            <Glyph name="mic" className="mr-4" /> Spoken
          </Tag>
        ) : null}
      </div>
      <dl className="border-t border-hairline">
        <Entry label="Why">
          <p className="font-medium">{receipt.why}</p>
        </Entry>
        <Entry label="Asked">
          <p className="whitespace-pre-wrap break-words">{receipt.asked}</p>
        </Entry>
        <Entry label="Did">
          {receipt.steps.length === 0 ? (
            <p className="text-stone">No tool ran.</p>
          ) : (
            <ul className="flex flex-col gap-4 font-mono text-caption">
              {receipt.steps.map((step, index) => (
                <li key={index} className="flex items-start gap-8">
                  <span className={`flex h-20 items-center ${OUTCOME[step.outcome].tone}`}>
                    <Glyph name={OUTCOME[step.outcome].glyph} />
                  </span>
                  <span className="min-w-0 break-words">
                    {TOOL_LABELS[step.name]?.done ?? step.name}
                    {step.summary ? <span className="text-stone"> · {step.summary}</span> : null}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </Entry>
        {receipt.searches.length > 0 ? (
          <Entry label="Searched">
            <ul className="flex flex-col gap-4 font-mono text-caption">
              {receipt.searches.map((search, index) => (
                <li key={index} className="break-words">
                  “{search.query}”
                  <span className="text-stone">
                    {" "}
                    · {BAND_LABELS[search.band]}
                    {search.best ? ` · best: ${search.best}` : ""}
                    {search.score != null ? ` (${search.score.toFixed(2)})` : ""}
                  </span>
                </li>
              ))}
            </ul>
          </Entry>
        ) : null}
        {receipt.check ? (
          <Entry label="Check">
            <p className="flex items-start gap-8">
              <span className={`flex h-20 items-center ${receipt.check === "grounded" ? "text-terminal-green" : "text-stone"}`}>
                <Glyph name={receipt.check === "grounded" ? "check" : receipt.check === "ungrounded" ? "cross" : "minus"} />
              </span>
              {CHECKS[receipt.check]}
            </p>
            {receipt.unsupported.length > 0 ? (
              <ul className="mt-8 flex list-disc flex-col gap-4 pl-24 text-caption text-charcoal">
                {receipt.unsupported.map((statement, index) => (
                  <li key={index}>{statement}</li>
                ))}
              </ul>
            ) : null}
          </Entry>
        ) : null}
        <Entry label="Told">
          <p className="whitespace-pre-wrap break-words">{receipt.told || <span className="text-stone">Nothing reached the customer.</span>}</p>
          {receipt.heard !== null ? (
            <p className="mt-8 font-mono text-caption text-stone">The customer cut the reply off. They heard: “{receipt.heard}”</p>
          ) : null}
        </Entry>
      </dl>
    </Card>
  );
}

function Transcript({ before, after, now }: { before: string[]; after: Line[]; now: number }) {
  const [earlier, setEarlier] = useState(false);
  if (before.length === 0 && after.length === 0) return null;
  return (
    <Card>
      <div className="flex items-center gap-8 px-16 py-12">
        <Caption>Conversation since the hand-off</Caption>
        {before.length > 0 ? (
          <Button variant="quiet" size="sm" className="ml-auto" aria-expanded={earlier} onClick={() => setEarlier((shown) => !shown)}>
            {earlier ? "Hide what came before" : "Show what came before"}
          </Button>
        ) : null}
      </div>
      {earlier ? (
        <ul className="flex flex-col gap-4 border-t border-hairline bg-paper-white px-16 py-12 font-mono text-label text-stone">
          {before.map((line, index) => (
            <li key={index} className="break-words">
              {line}
            </li>
          ))}
        </ul>
      ) : null}
      <ul className="flex flex-col border-t border-hairline">
        {after.map((line, index) => (
          <li key={index} className="grid grid-cols-[96px_minmax(0,1fr)] gap-12 border-t border-hairline px-16 py-8 first:border-t-0 max-sm:grid-cols-1 max-sm:gap-2">
            <span className="label-mono flex items-center gap-4 pt-2 text-stone">
              {line.role === "customer" ? "Customer" : "Assistant"}
              {line.mode === "voice" ? <Glyph name="mic" /> : null}
            </span>
            <span className="min-w-0">
              <span className="whitespace-pre-wrap break-words text-body-sm text-obsidian">{line.text}</span>{" "}
              <span className="whitespace-nowrap font-mono text-label text-smoke">{formatWhen(line.at, now)}</span>
            </span>
          </li>
        ))}
      </ul>
    </Card>
  );
}

/** A change of hands, status or priority, as one line. */
function activityLine(item: Activity): string {
  switch (item.kind) {
    case "created":
      return "The assistant opened the ticket";
    case "assigned":
      return item.body ? (item.body === item.author ? `${item.author} took the ticket` : `${item.author} gave the ticket to ${item.body}`) : `${item.author} put the ticket back in the queue`;
    case "status":
      return `${item.author} set it to ${(STATUS_LABELS[item.body as DeskTicket["status"]] ?? item.body).toLowerCase()}`;
    case "priority":
      return `${item.author} set the priority to ${item.body}`;
    default:
      return item.author;
  }
}

function Timeline({ activity, now }: { activity: Activity[]; now: number }) {
  return (
    <ol className="flex flex-col gap-12">
      {activity.map((item) => {
        const when = <span className="whitespace-nowrap font-mono text-label text-smoke">{formatWhen(item.at, now)}</span>;
        if (item.kind === "reply" || item.kind === "refund") {
          return (
            <li key={item.id} className="flex flex-col gap-8 rounded-md bg-obsidian p-16 text-pure-white">
              <p className="label-mono flex flex-wrap items-center gap-8 text-ash">
                <Glyph name={item.kind === "refund" ? "card" : "reply"} />
                {item.kind === "refund" ? "Refund approved" : "Reply to the customer"}
                <span className="text-smoke">· {item.author}</span>
                <span className="ml-auto font-mono text-label normal-case tracking-normal text-smoke">{formatWhen(item.at, now)}</span>
              </p>
              <p className="whitespace-pre-wrap break-words text-body-sm">{item.body}</p>
            </li>
          );
        }
        if (item.kind === "note") {
          return (
            <li key={item.id} className="flex flex-col gap-8 rounded-md border border-dashed border-ash bg-paper-white p-16">
              <p className="label-mono flex flex-wrap items-center gap-8 text-stone">
                <Glyph name="note" />
                Internal note
                <span className="text-smoke">· {item.author}</span>
                <span className="ml-auto font-mono text-label normal-case tracking-normal text-smoke">{formatWhen(item.at, now)}</span>
              </p>
              <p className="whitespace-pre-wrap break-words text-body-sm text-obsidian">{item.body}</p>
            </li>
          );
        }
        return (
          <li key={item.id} className="flex items-center gap-8 px-16 font-mono text-label text-stone">
            <Glyph name={item.kind === "created" ? "dot" : item.kind === "assigned" ? "person" : item.kind === "status" ? "swap" : "minus"} />
            <span className="min-w-0 flex-1 break-words">
              {activityLine(item)}
              {item.public ? "" : " · not shown to the customer"}
            </span>
            {when}
          </li>
        );
      })}
    </ol>
  );
}

/** Replies worth keeping at hand. They are a start: the person sending them edits them. */
function savedReplies(name: string): { label: string; text: string }[] {
  return [
    { label: "Looking into it", text: `Hi ${name}, thanks for waiting. I have your case and I am looking into it now. I will write here as soon as I know more.` },
    { label: "Refund approved", text: `Hi ${name}, I have approved the refund. It goes back to your original payment method and usually shows within 5 to 10 business days.` },
    { label: "Need more detail", text: `Hi ${name}, could you tell me a bit more about what happened, and when? With that I can sort this out for you.` },
  ];
}

type Draft = { mode: "reply" | "note"; text: string };

/**
 * Where a person answers. A reply goes into the customer's conversation. A note stays at the
 * desk. The two look different on purpose: sending the wrong one is the mistake to prevent.
 */
function Answer({ ticket, desk, online, name }: { ticket: DeskTicket; desk: Desk; online: boolean; name: string }) {
  const [draft, setDraft] = useState<Draft>({ mode: "reply", text: "" });
  // What was last sent, so a refusal can give the words back. `seen` is the refusal handled.
  const [sent, setSent] = useState<Draft | null>(null);
  const refusal = desk.state.refusal?.key === ticket.key ? desk.state.refusal : null;
  const [seen, setSeen] = useState(refusal?.n ?? 0);
  if (refusal && refusal.n !== seen) {
    setSeen(refusal.n);
    if (sent && draft.text === "") setDraft(sent);
  }
  const text = draft.text.trim();
  const send = (status?: DeskTicket["status"]) => {
    if (!text || !online) return;
    if (draft.mode === "note") desk.note(ticket.key, text);
    else desk.reply(ticket.key, text, status);
    setSent(draft);
    setDraft({ mode: draft.mode, text: "" });
  };
  const note = draft.mode === "note";
  return (
    <Card className={`flex flex-col ${note ? "border border-dashed border-ash bg-paper-white shadow-none" : ""}`}>
      <div className="flex flex-wrap items-center gap-4 border-b border-hairline px-12 py-8" role="tablist" aria-label="What to write">
        {(["reply", "note"] as const).map((mode) => (
          <button
            key={mode}
            type="button"
            role="tab"
            aria-selected={draft.mode === mode}
            onClick={() => setDraft((before) => ({ ...before, mode }))}
            className={`inline-flex h-28 items-center gap-8 rounded-md px-8 text-caption transition-colors ${
              draft.mode === mode ? "bg-hairline text-obsidian" : "text-stone hover:text-obsidian"
            }`}
          >
            <Glyph name={mode === "reply" ? "reply" : "note"} />
            {mode === "reply" ? "Reply to the customer" : "Internal note"}
          </button>
        ))}
        {!note ? (
          <div className="ml-auto flex flex-wrap items-center gap-4">
            {savedReplies(name).map((saved) => (
              <Button key={saved.label} variant="quiet" size="sm" onClick={() => setDraft({ mode: "reply", text: saved.text })}>
                {saved.label}
              </Button>
            ))}
          </div>
        ) : null}
      </div>
      <label className="flex flex-col">
        <span className="sr-only">{note ? "Internal note" : "Reply to the customer"}</span>
        <textarea
          value={draft.text}
          rows={4}
          maxLength={2000}
          onChange={(event) => setDraft((before) => ({ ...before, text: event.target.value }))}
          onKeyDown={(event) => {
            if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) send();
          }}
          placeholder={note ? "Write a note for the team. The customer never sees it." : "Write to the customer. It appears in their conversation at once."}
          className="min-h-[96px] resize-y bg-transparent px-16 py-12 text-body-sm text-obsidian outline-none placeholder:text-smoke"
        />
      </label>
      <div className="flex flex-wrap items-center gap-8 border-t border-hairline px-12 py-8">
        <p className="mr-auto font-mono text-label text-stone">
          {note ? "Stays at the desk." : ticket.assignee ? "Goes into the customer's conversation." : "Goes into the customer's conversation, and the ticket becomes yours."}
        </p>
        {note ? (
          <Button onClick={() => send()} disabled={!text || !online}>
            Add note
          </Button>
        ) : (
          <>
            <Button onClick={() => send("resolved")} disabled={!text || !online}>
              Reply and resolve
            </Button>
            <Button variant="primary" onClick={() => send()} disabled={!text || !online}>
              Send reply
            </Button>
          </>
        )}
      </div>
    </Card>
  );
}

function InvoiceRow({ invoice, refundable, online, onRefund }: { invoice: AccountInvoice; refundable: boolean; online: boolean; onRefund: () => void }) {
  const [asking, setAsking] = useState(false);
  return (
    <li className="flex flex-col gap-8 border-t border-hairline px-16 py-12">
      <div className="flex items-baseline gap-8">
        <span className="font-mono text-caption text-obsidian">{invoice.id}</span>
        <span className="ml-auto font-mono text-caption tabular-nums text-obsidian">{formatUsd(invoice.amount_cents)}</span>
      </div>
      <div className="flex items-center gap-8">
        <span className="min-w-0 flex-1 truncate text-caption text-stone" title={invoice.description}>
          {invoice.issued_on} · {invoice.description}
        </span>
        {invoice.status === "paid" && refundable && !asking ? (
          <Button size="sm" disabled={!online} onClick={() => setAsking(true)}>
            Refund
          </Button>
        ) : (
          <Tag tone={invoice.status === "refunded" ? "good" : "plain"}>{invoice.status}</Tag>
        )}
      </div>
      {asking && invoice.status === "paid" ? (
        <div className="flex flex-col gap-8 rounded-md bg-paper-white p-12 shadow-hairline">
          <p className="text-caption text-charcoal">
            Refund {formatUsd(invoice.amount_cents)} to the customer? The billing team may go above the assistant&apos;s limit. The refund
            window and the once-only rule still hold.
          </p>
          <div className="flex gap-8">
            <Button
              size="sm"
              variant="surface"
              disabled={!online}
              onClick={() => {
                setAsking(false);
                onRefund();
              }}
            >
              Approve refund
            </Button>
            <Button size="sm" variant="quiet" onClick={() => setAsking(false)}>
              Cancel
            </Button>
          </div>
        </div>
      ) : null}
    </li>
  );
}

function Fact({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-baseline gap-12">
      <dt className="label-mono w-[88px] shrink-0 text-stone">{label}</dt>
      <dd className="min-w-0 break-words font-mono text-caption text-obsidian">{children}</dd>
    </div>
  );
}

/** The customer behind the ticket, from the conversation's own copy of the CRM. */
function Account({ account, online, onRefund }: { account: AccountView; online: boolean; onRefund: (invoiceId: string) => void }) {
  return (
    <>
      <Card className="flex flex-col gap-12 p-16">
        <Caption>Customer</Caption>
        <div className="flex flex-col gap-4">
          <p className="text-body font-medium text-obsidian">{account.company}</p>
          <p className="text-body-sm text-charcoal">{account.contact}</p>
        </div>
        <div className="flex flex-wrap gap-8">
          <Tag>
            {PLAN_LABELS[account.tier]}
            {account.billing_cycle ? ` · ${account.billing_cycle}` : ""}
          </Tag>
          <Tag tone={account.status === "active" ? "good" : "solid"}>{account.status.replace("_", " ")}</Tag>
        </div>
        <dl className="flex flex-col gap-4">
          <Fact label="Account">{account.account_number}</Fact>
          <Fact label="Email">{account.email}</Fact>
          <Fact label="Billing">{account.billing_email}</Fact>
          <Fact label="Phone">{account.phone}</Fact>
          <Fact label="Seats">{account.seats}</Fact>
          <Fact label="Since">{account.customer_since}</Fact>
          {account.renews_on ? <Fact label="Renews">{account.renews_on}</Fact> : null}
        </dl>
      </Card>
      <Card className="flex flex-col">
        <Caption className="px-16 py-12">Invoices</Caption>
        {account.invoices.length === 0 ? (
          <p className="border-t border-hairline px-16 py-12 text-caption text-stone">This account has never been charged.</p>
        ) : (
          <ul className="flex flex-col">
            {account.invoices.map((invoice) => (
              <InvoiceRow key={invoice.id} invoice={invoice} refundable={account.tier !== "free"} online={online} onRefund={() => onRefund(invoice.id)} />
            ))}
          </ul>
        )}
      </Card>
      {account.refunds.length > 0 ? (
        <Card className="flex flex-col">
          <Caption className="px-16 py-12">Refunds</Caption>
          <ul className="flex flex-col">
            {account.refunds.map((refund) => (
              <li key={refund.id} className="flex flex-col gap-4 border-t border-hairline px-16 py-12">
                <div className="flex items-baseline gap-8">
                  <span className="flex items-center gap-4 font-mono text-caption text-obsidian">
                    <Glyph name="check" className="text-terminal-green" /> {refund.id}
                  </span>
                  <span className="ml-auto font-mono text-caption tabular-nums text-obsidian">{formatUsd(refund.amount_cents)}</span>
                </div>
                <span className="text-caption text-stone">
                  {refund.refunded_on} · {refund.invoice_id} · {refund.reason.replace(/_/g, " ")}
                </span>
              </li>
            ))}
          </ul>
        </Card>
      ) : null}
      {account.changes.length > 0 ? (
        <Card className="flex flex-col">
          <Caption className="px-16 py-12">Changes in this conversation</Caption>
          <ul className="flex flex-col">
            {account.changes.map((change, index) => (
              <li key={index} className="break-words border-t border-hairline px-16 py-12 font-mono text-label text-stone">
                <span className="text-obsidian">{change.field.replace(/_/g, " ")}</span> · {change.old_value} → <span className="text-obsidian">{change.new_value}</span>
              </li>
            ))}
          </ul>
        </Card>
      ) : null}
    </>
  );
}

interface Props {
  desk: Desk;
  ticket: DeskTicket;
  detail: DeskDetail | null; // null until the server has sent the case
  now: number;
  onBack: () => void;
}

export function DeskCase({ desk, ticket, detail, now, onBack }: Props) {
  const online = desk.state.connection === "open";
  const me = desk.state.agent;
  const refusal = desk.state.refusal?.key === ticket.key ? desk.state.refusal : null;
  const firstName = (ticket.contact ?? "").split(" ")[0] || "there";
  const opened = formatAgo(ticket.created_at, now);
  return (
    <div className="mx-auto flex w-full max-w-[1120px] flex-col gap-16 p-24 max-sm:p-16">
      <header className="flex flex-col gap-12">
        <div className="flex flex-wrap items-center gap-8">
          <Button variant="quiet" size="sm" onClick={onBack} className="-ml-8 lg:hidden">
            <Glyph name="left" /> Tickets
          </Button>
          <span className="font-mono text-body-sm font-medium text-obsidian">{ticket.id}</span>
          <Tag tone={ticket.status === "resolved" ? "good" : ticket.status === "open" ? "solid" : "plain"}>
            {ticket.status === "resolved" ? <Glyph name="check" className="mr-4" /> : null}
            {STATUS_LABELS[ticket.status]}
          </Tag>
          <Tag>{ticket.priority} priority</Tag>
          {ticket.channel === "voice" ? (
            <Tag>
              <Glyph name="mic" className="mr-4" /> Voice
            </Tag>
          ) : null}
          <span className="ml-auto font-mono text-label text-stone">opened {opened === "now" ? "just now" : `${opened} ago`}</span>
        </div>
        <h2 className="text-heading font-w450 text-carbon max-sm:text-[24px]">{TICKET_REASONS[ticket.reason]}</h2>
        <p className="max-w-[720px] whitespace-pre-wrap break-words text-body-sm text-charcoal">{ticket.summary}</p>
        <div className="flex flex-wrap items-center gap-8">
          {ticket.assignee === me ? (
            <Button size="sm" disabled={!online} onClick={() => desk.assign(ticket.key, null)}>
              <Glyph name="person" /> Yours · Put back
            </Button>
          ) : (
            <Button size="sm" disabled={!online} onClick={() => desk.assign(ticket.key, me)}>
              <Glyph name="person" /> {ticket.assignee ? `Take over from ${ticket.assignee}` : "Assign to me"}
            </Button>
          )}
          <div className="flex h-28 items-center rounded-md bg-pure-white p-2 shadow-hairline" role="group" aria-label="Status">
            {STATUSES.map((status) => (
              <button
                key={status}
                type="button"
                aria-pressed={ticket.status === status}
                disabled={!online}
                onClick={() => ticket.status !== status && desk.setStatus(ticket.key, status)}
                className={`inline-flex h-24 items-center rounded-sm px-8 text-caption transition-colors disabled:opacity-40 ${
                  ticket.status === status ? "bg-hairline text-obsidian" : "text-stone hover:text-obsidian"
                }`}
              >
                {STATUS_LABELS[status]}
              </button>
            ))}
          </div>
          <label className="flex h-28 items-center gap-4 rounded-md bg-pure-white pl-8 text-caption text-stone shadow-hairline">
            Priority
            <select
              value={ticket.priority}
              disabled={!online}
              onChange={(event) => desk.setPriority(ticket.key, event.target.value as DeskTicket["priority"])}
              className="h-28 cursor-pointer rounded-md bg-transparent pl-4 pr-8 text-caption text-obsidian outline-none"
            >
              {PRIORITIES.map((priority) => (
                <option key={priority} value={priority}>
                  {priority}
                </option>
              ))}
            </select>
          </label>
        </div>
      </header>

      {refusal ? (
        <div role="alert" className="flex items-start gap-8 rounded-md bg-pure-white px-16 py-12 text-body-sm text-obsidian shadow-ring">
          <span className="flex h-20 items-center">
            <Glyph name="alert" />
          </span>
          <p className="min-w-0 flex-1">{refusal.message}</p>
          <Button variant="quiet" size="sm" onClick={desk.dismiss}>
            Dismiss
          </Button>
        </div>
      ) : null}

      {detail === null ? (
        <p className="flex items-center gap-8 py-32 font-mono text-caption text-stone" role="status">
          <Spinner /> Loading the case…
        </p>
      ) : (
        <div className="grid grid-cols-[minmax(0,1fr)_340px] items-start gap-16 max-xl:grid-cols-1">
          <div className="flex min-w-0 flex-col gap-16">
            {detail.receipt ? <ReceiptCard receipt={detail.receipt} /> : null}
            <Transcript before={detail.before} after={detail.after} now={now} />
            <section className="flex flex-col gap-12" aria-label="Activity">
              <Caption className="px-16">Activity</Caption>
              <Timeline activity={detail.activity} now={now} />
            </section>
            <Answer ticket={ticket} desk={desk} online={online} name={firstName} />
          </div>
          <aside className="flex min-w-0 flex-col gap-16" aria-label="Customer">
            {detail.account ? (
              <Account account={detail.account} online={online} onRefund={(invoiceId) => desk.refund(ticket.key, invoiceId)} />
            ) : (
              <Card className="flex flex-col gap-8 p-16">
                <Caption>Customer</Caption>
                <p className="text-body-sm text-charcoal">Nobody was verified in this conversation, so there is no account to show.</p>
              </Card>
            )}
          </aside>
        </div>
      )}
    </div>
  );
}
