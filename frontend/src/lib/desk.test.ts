import { describe, expect, it } from "vitest";

import { type DeskState, deskReduce, initialDesk } from "./desk";
import type { DeskDetail, DeskEvent, DeskStats, DeskTicket } from "./protocol";

const stats = (open: number): DeskStats => ({ open, pending: 0, resolved: 0, unassigned: open, conversations: 1, turns: 2, handed_off: open, first_reply_s: null });

const ticket = (key: string, extra: Partial<DeskTicket> = {}): DeskTicket => ({
  key,
  id: `ESC-${key.toUpperCase()}`,
  status: "open",
  priority: "normal",
  reason: "customer_request",
  summary: "Wants a person.",
  channel: "text",
  assignee: null,
  account_number: null,
  company: null,
  contact: null,
  tier: null,
  created_at: 100,
  updated_at: 100,
  replies: 0,
  ...extra,
});

const detail = (of: DeskTicket): DeskDetail => ({ type: "desk.detail", ticket: of, activity: [], receipt: null, before: [], after: [], account: null });

function fold(...events: DeskEvent[]): DeskState {
  return events.reduce((state, event) => deskReduce(state, { type: "event", event }), initialDesk);
}

const ready = (...tickets: DeskTicket[]): DeskEvent => ({ type: "desk.ready", agent: "Sam Carter", tickets, stats: stats(tickets.length) });

describe("the desk in the page", () => {
  it("lists a ticket the moment it is opened, newest first, with the workspace's numbers", () => {
    const state = fold(ready(ticket("a", { created_at: 100 })), { type: "desk.ticket", ticket: ticket("b", { created_at: 200 }), stats: stats(2) });
    expect(state.connection).toBe("open");
    expect(state.tickets.map((t) => t.key)).toEqual(["b", "a"]);
    expect(state.stats?.open).toBe(2);
  });

  it("replaces a ticket that changed instead of listing it twice", () => {
    const state = fold(ready(ticket("a")), { type: "desk.ticket", ticket: ticket("a", { status: "resolved", assignee: "Priya" }), stats: stats(0) });
    expect(state.tickets).toHaveLength(1);
    expect([state.tickets[0].status, state.tickets[0].assignee]).toEqual(["resolved", "Priya"]);
  });

  it("lets the header of an open case follow its ticket before the case is loaded again", () => {
    const state = fold(ready(ticket("a")), detail(ticket("a")), { type: "desk.ticket", ticket: ticket("a", { assignee: "Priya" }), stats: stats(1) });
    expect(state.details.a.ticket.assignee).toBe("Priya");
  });

  it("forgets the cases it held when the desk connects again, and keeps what the server turned down", () => {
    const refused: DeskEvent = { type: "desk.error", code: "refused", message: "This invoice was already refunded.", key: "a" };
    const state = fold(ready(ticket("a")), detail(ticket("a")), refused, refused, ready(ticket("a")));
    expect(state.details).toEqual({});
    expect(state.refusal).toEqual({ key: "a", message: "This invoice was already refunded.", n: 2 });
    expect(deskReduce(state, { type: "dismiss" }).refusal).toBeNull();
  });
});
