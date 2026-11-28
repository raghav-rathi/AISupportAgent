"use client";

import { useEffect, useMemo, useState } from "react";

import type { Desk as DeskLink } from "@/lib/desk";
import { PLAN_LABELS, STATUS_LABELS, TICKET_REASONS, formatAgo, formatSpan } from "@/lib/labels";
import type { DeskStats, DeskTicket } from "@/lib/protocol";

import { DeskCase } from "./DeskCase";
import { Button, Caption, Card, Eyebrow, Glyph, Tag } from "./ui";

/** The time, in seconds, moving often enough for "4m ago" to stay true. */
function useNow(): number {
  const [now, setNow] = useState(() => Date.now() / 1000);
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now() / 1000), 20_000);
    return () => window.clearInterval(timer);
  }, []);
  return now;
}

type Filter = DeskTicket["status"] | "all";

const FILTERS: { filter: Filter; label: string }[] = [
  { filter: "open", label: "Open" },
  { filter: "pending", label: "Waiting" },
  { filter: "resolved", label: "Resolved" },
  { filter: "all", label: "All" },
];

const URGENCY: Record<DeskTicket["priority"], number> = { urgent: 0, high: 1, normal: 2, low: 3 };

function Figure({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div className="flex min-w-[148px] shrink-0 flex-col gap-4 border-l border-hairline p-16 first:border-l-0">
      <dt className="label-mono text-stone">{label}</dt>
      <dd className="text-heading font-w450 tabular-nums text-carbon">{value}</dd>
      <dd className="font-mono text-label text-stone">{note}</dd>
    </div>
  );
}

/** The workspace in five numbers. The fourth is the one the assistant is there for. */
function Figures({ stats }: { stats: DeskStats }) {
  const alone = stats.turns - stats.handed_off;
  return (
    <Card>
      <dl className="grid grid-cols-5 max-lg:flex max-lg:overflow-x-auto">
        <Figure label="Open" value={String(stats.open)} note={`${stats.unassigned} unassigned`} />
        <Figure label="Waiting on customer" value={String(stats.pending)} note="replied, not resolved" />
        <Figure label="Resolved" value={String(stats.resolved)} note="closed by a person" />
        <Figure
          label="Settled by the assistant"
          value={stats.turns > 0 ? `${Math.round((100 * alone) / stats.turns)}%` : "n/a"}
          note={`${alone} of ${stats.turns} turns`}
        />
        <Figure
          label="First reply, median"
          value={formatSpan(stats.first_reply_s)}
          note={`${stats.conversations} ${stats.conversations === 1 ? "conversation" : "conversations"}`}
        />
      </dl>
    </Card>
  );
}

function Row({ ticket, now, selected, mine, onSelect }: { ticket: DeskTicket; now: number; selected: boolean; mine: boolean; onSelect: () => void }) {
  const who = ticket.company ?? "Not verified";
  return (
    <li className="border-b border-hairline">
      <button
        type="button"
        aria-current={selected ? "true" : undefined}
        onClick={onSelect}
        className={`flex w-full flex-col gap-4 border-l-2 px-16 py-12 text-left transition-colors ${
          selected ? "border-obsidian bg-pure-white" : "border-transparent hover:bg-pure-white/70"
        }`}
      >
        <span className="flex items-center gap-8">
          <span className="font-mono text-label text-obsidian">{ticket.id}</span>
          {ticket.status === "resolved" ? (
            <Glyph name="check" className="text-terminal-green" />
          ) : ticket.priority === "urgent" || ticket.priority === "high" ? (
            <Tag tone={ticket.priority === "urgent" ? "solid" : "plain"}>{ticket.priority}</Tag>
          ) : null}
          {ticket.channel === "voice" ? <Glyph name="mic" className="text-stone" /> : null}
          <span className="ml-auto font-mono text-label tabular-nums text-smoke">{formatAgo(ticket.created_at, now)}</span>
        </span>
        <span className={`text-body-sm ${ticket.status === "resolved" ? "text-stone" : "font-medium text-obsidian"}`}>
          {TICKET_REASONS[ticket.reason]}
        </span>
        <span className="line-clamp-2 text-caption text-stone">{ticket.summary}</span>
        <span className="flex items-center gap-8 font-mono text-label text-stone">
          <span className="truncate">
            {who}
            {ticket.tier ? ` · ${PLAN_LABELS[ticket.tier]}` : ""}
          </span>
          <span className="ml-auto shrink-0">
            {ticket.assignee ? (mine ? "You" : ticket.assignee) : "Unassigned"}
            {ticket.replies > 0 ? ` · ${ticket.replies} ${ticket.replies === 1 ? "reply" : "replies"}` : ""}
          </span>
        </span>
      </button>
    </li>
  );
}

function AgentName({ agent, onRename }: { agent: string; onRename: (agent: string) => void }) {
  const [draft, setDraft] = useState<string | null>(null); // null while nobody is typing
  const commit = () => {
    const name = draft?.trim();
    setDraft(null);
    if (name && name !== agent) onRename(name);
  };
  return (
    <label className="flex items-center gap-8">
      <span className="label-mono text-stone">Acting as</span>
      <input
        value={draft ?? agent}
        maxLength={40}
        onChange={(event) => setDraft(event.target.value)}
        onBlur={commit}
        onKeyDown={(event) => {
          if (event.key === "Enter") event.currentTarget.blur();
        }}
        className="h-28 w-[152px] rounded-md bg-pure-white px-8 text-caption text-obsidian shadow-hairline"
      />
    </label>
  );
}

/** Messages that end in a ticket, for a desk that has none to show yet. */
const OPENERS = [
  { label: "Ask for a person", text: "I'd like to talk to a person about a contract question." },
  { label: "Refund above the limit", text: "My account number is 20573846 and my PIN is 9034. Please refund invoice INV-30388." },
];

function Empty({ onTry }: { onTry: (text: string) => void }) {
  return (
    <div className="mx-auto flex w-full max-w-520 flex-col gap-16 px-24 py-64">
      <Eyebrow>No tickets yet</Eyebrow>
      <p className="text-heading font-w450 text-carbon">Nothing has been handed to a person.</p>
      <p className="text-body-sm text-charcoal">
        A ticket lands here the moment the assistant gives a conversation to a person: a refund above its limit, a question the help
        center does not answer, a customer who asks for someone. Each one arrives with a record of what the assistant tried.
      </p>
      <div className="flex flex-wrap gap-8">
        {OPENERS.map((opener) => (
          <Button key={opener.label} onClick={() => onTry(opener.text)}>
            {opener.label} <Glyph name="right" />
          </Button>
        ))}
      </div>
    </div>
  );
}

interface Props {
  desk: DeskLink;
  selected: string | null; // the key of the ticket whose case is shown
  onSelect: (key: string | null) => void;
  /** Send a message in the assistant's view, for a desk with nothing to show. */
  onTry: (text: string) => void;
}

export function Desk({ desk, selected, onSelect, onTry }: Props) {
  const { state } = desk;
  const now = useNow();
  const [chosen, setFilter] = useState<Filter | null>(null);
  const [onlyMine, setOnlyMine] = useState(false);
  const [query, setQuery] = useState("");
  const { open } = desk;

  // The server keeps the case on screen fresh for as long as the desk says which one it is.
  useEffect(() => {
    open(selected);
    return () => open(null);
  }, [open, selected]);

  const counts = useMemo(() => {
    const by: Record<Filter, number> = { open: 0, pending: 0, resolved: 0, all: state.tickets.length };
    for (const ticket of state.tickets) by[ticket.status] += 1;
    return by;
  }, [state.tickets]);

  // Until someone chooses, the list shows what needs a person, or everything when nothing does.
  const filter = chosen ?? (counts.open > 0 ? "open" : "all");
  const needle = query.trim().toLowerCase();
  const listed = state.tickets
    .filter((ticket) => filter === "all" || ticket.status === filter)
    .filter((ticket) => !onlyMine || ticket.assignee === state.agent)
    .filter(
      (ticket) =>
        !needle ||
        [ticket.id, ticket.company, ticket.contact, ticket.account_number, ticket.summary, TICKET_REASONS[ticket.reason], ticket.assignee]
          .join(" ")
          .toLowerCase()
          .includes(needle),
    )
    // What needs someone first: the urgent before the rest, and the oldest of those first.
    .sort((a, b) => (filter === "open" ? URGENCY[a.priority] - URGENCY[b.priority] || a.created_at - b.created_at : b.updated_at - a.updated_at));

  const shown = selected ? (state.tickets.find((ticket) => ticket.key === selected) ?? null) : null;
  const online = state.connection === "open";

  return (
    <main className="flex min-h-0 flex-1 flex-col">
      <div className={`flex shrink-0 flex-col gap-16 px-24 pb-16 pt-8 max-sm:px-16 ${shown ? "max-lg:hidden" : ""}`}>
        <div className="flex flex-wrap items-end justify-between gap-16">
          <div className="flex flex-col gap-8">
            <Eyebrow>Service desk</Eyebrow>
            <h1 className="text-heading font-w450 text-carbon">Tickets</h1>
          </div>
          <AgentName agent={state.agent} onRename={desk.rename} />
        </div>
        {state.stats ? <Figures stats={state.stats} /> : null}
      </div>

      {state.tickets.length === 0 ? (
        <div className="relative min-h-0 flex-1 overflow-y-auto border-t border-hairline">
          {online ? (
            <Empty onTry={onTry} />
          ) : (
            <p className="px-24 py-64 text-center font-mono text-caption text-stone" role="status">
              {state.connection === "connecting" ? "Connecting to the desk…" : "The desk is not connected. Trying again…"}
            </p>
          )}
        </div>
      ) : (
        <div className="flex min-h-0 flex-1 border-t border-hairline">
          <section
            aria-label="Tickets"
            className={`flex w-[360px] shrink-0 flex-col border-r border-hairline max-lg:w-full max-lg:border-r-0 ${shown ? "max-lg:hidden" : ""}`}
          >
            <div className="flex flex-col gap-8 border-b border-hairline p-12">
              <div className="flex items-center gap-4" role="tablist" aria-label="Status">
                {FILTERS.map((item) => (
                  <button
                    key={item.filter}
                    type="button"
                    role="tab"
                    aria-selected={filter === item.filter}
                    onClick={() => setFilter(item.filter)}
                    className={`inline-flex h-28 items-center gap-4 rounded-md px-8 text-caption transition-colors ${
                      filter === item.filter ? "bg-hairline text-obsidian" : "text-stone hover:text-obsidian"
                    }`}
                  >
                    {item.label}
                    <span className="font-mono text-eyebrow tabular-nums">{counts[item.filter]}</span>
                  </button>
                ))}
                <button
                  type="button"
                  aria-pressed={onlyMine}
                  onClick={() => setOnlyMine((before) => !before)}
                  className={`ml-auto inline-flex h-28 items-center rounded-md px-8 text-caption transition-colors ${
                    onlyMine ? "bg-hairline text-obsidian" : "text-stone hover:text-obsidian"
                  }`}
                >
                  Mine
                </button>
              </div>
              <label className="flex h-32 items-center gap-8 rounded-md bg-pure-white px-8 text-stone shadow-hairline focus-within:shadow-focus">
                <Glyph name="search" />
                <span className="sr-only">Search tickets</span>
                <input
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Search by ID, company or words"
                  className="min-w-0 flex-1 bg-transparent text-caption text-obsidian outline-none placeholder:text-smoke"
                />
              </label>
            </div>
            {listed.length === 0 ? (
              <p className="px-16 py-32 text-center font-mono text-label text-stone">
                {needle || onlyMine ? "No ticket matches." : `No ticket is ${filter === "all" ? "here" : STATUS_LABELS[filter as DeskTicket["status"]].toLowerCase()}.`}
              </p>
            ) : (
              <ul className="relative min-h-0 flex-1 overflow-y-auto">
                {listed.map((ticket) => (
                  <Row
                    key={ticket.key}
                    ticket={ticket}
                    now={now}
                    selected={ticket.key === selected}
                    mine={ticket.assignee === state.agent}
                    onSelect={() => onSelect(ticket.key)}
                  />
                ))}
              </ul>
            )}
          </section>

          <section aria-label="Case" className={`relative min-w-0 flex-1 overflow-y-auto ${shown ? "" : "max-lg:hidden"}`}>
            {shown ? (
              <DeskCase key={shown.key} desk={desk} ticket={shown} detail={state.details[shown.key] ?? null} now={now} onBack={() => onSelect(null)} />
            ) : (
              <div className="flex h-full flex-col items-center justify-center gap-8 p-24 text-center">
                <Glyph name="inbox" className="text-smoke" />
                <Caption>Choose a ticket</Caption>
                <p className="max-w-[320px] text-caption text-stone">
                  Each one opens with what the assistant did before it stopped, the customer&apos;s account, and a place to answer.
                </p>
              </div>
            )}
          </section>
        </div>
      )}
    </main>
  );
}
