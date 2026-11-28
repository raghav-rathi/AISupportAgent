/** The words the UI uses for what the backend reports, and how it prints numbers. */

import type { CaseUpdated, DeskTicket, RetrievalResult, Ticket } from "./protocol";
import type { FailureReason } from "./state";

/** Each tool, while it runs and once it is done. */
export const TOOL_LABELS: Record<string, { running: string; done: string }> = {
  verify_identity: { running: "Verifying identity, plan and contract status", done: "Identity check" },
  get_account: { running: "Reading the account", done: "Account" },
  search_knowledge_base: { running: "Searching the knowledge base", done: "Knowledge base" },
  issue_refund: { running: "Issuing the refund", done: "Refund" },
  update_account_field: { running: "Updating the account", done: "Account update" },
  create_escalation_ticket: { running: "Opening a ticket for a specialist", done: "Escalation ticket" },
  get_ticket_status: { running: "Checking what the support team has said", done: "Ticket status" },
};

export const BAND_LABELS: Record<RetrievalResult["band"], string> = {
  answer: "Confident match",
  check: "Partial match, so the answer is held for a check",
  verify: "Covered for a plan, so identity is needed first",
  restricted: "Covered for another plan",
  escalate: "No answer in the help center",
};

export const PLAN_LABELS: Record<Ticket["tier"] & string, string> = { free: "Free", pro: "Pro", enterprise: "Enterprise" };

/** What happened to a model call, as the end of "The model ...". */
export const FAILURE_LABELS: Record<FailureReason, string> = {
  timeout: "timed out",
  rate_limited: "was rate limited",
  server_error: "returned an error",
  connection: "could not be reached",
  quota: "refused the request",
  rejected: "rejected the request",
};

export const TICKET_REASONS: Record<Ticket["reason"], string> = {
  low_confidence: "No answer in the help center",
  ungrounded_answer: "Answer failed its grounding check",
  refund_over_limit: "Refund above the support limit",
  account_suspended: "Suspended account",
  verification_failed: "Identity check failed",
  billing_dispute: "Billing dispute",
  security: "Security",
  privacy_request: "Privacy request",
  customer_request: "Customer asked for a person",
  unanswered_question: "Question the help center does not answer",
  step_limit: "Step limit reached",
  other: "Other",
};

const whole = new Intl.NumberFormat("en-US");

export function formatMs(ms: number | null | undefined): string {
  if (ms == null) return "n/a";
  const whole = Math.round(ms);
  return whole < 1000 ? `${whole} ms` : `${(ms / 1000).toFixed(ms < 10_000 ? 2 : 1)} s`;
}

export function formatCount(n: number | null | undefined): string {
  return n == null ? "n/a" : whole.format(n);
}

/** A cost in dollars. Small amounts keep enough digits to be told apart. */
export function formatCost(usd: number | null | undefined): string {
  if (usd == null) return "unknown";
  if (usd === 0) return "$0";
  return `$${usd.toFixed(usd < 0.01 ? 5 : 4)}`;
}

/** The model's ID without its vendor: "google/gemma-4-31b-it" reads as "gemma-4-31b-it". */
export function shortModel(model: string): string {
  return model.slice(model.lastIndexOf("/") + 1);
}

/** Where a ticket stands, in the words of the desk. */
export const STATUS_LABELS: Record<DeskTicket["status"], string> = {
  open: "Open",
  pending: "Waiting on customer",
  resolved: "Resolved",
};

/** The same, as the customer reads it on their own ticket. */
export const CASE_LABELS: Record<CaseUpdated["status"], string> = {
  open: "With the support team",
  pending: "Waiting for your answer",
  resolved: "Resolved",
};

const dollars = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });

/** An amount kept in cents, as dollars: 144000 reads as "$1,440.00". */
export function formatUsd(cents: number): string {
  return dollars.format(cents / 100);
}

/** How long ago, short enough for a list: "now", "4m", "3h", "2d". Times are in seconds. */
export function formatAgo(at: number, now: number): string {
  const seconds = Math.max(0, now - at);
  if (seconds < 45) return "now";
  if (seconds < 3600) return `${Math.max(1, Math.round(seconds / 60))}m`;
  if (seconds < 86_400) return `${Math.round(seconds / 3600)}h`;
  return `${Math.round(seconds / 86_400)}d`;
}

/** A length of time for a figure: "38 s", "4 min", "2.5 h". */
export function formatSpan(seconds: number | null | undefined): string {
  if (seconds == null) return "n/a";
  if (seconds < 60) return `${Math.round(seconds)} s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)} min`;
  return `${(seconds / 3600).toFixed(1)} h`;
}

const clock = new Intl.DateTimeFormat("en-US", { hour: "numeric", minute: "2-digit" });
const day = new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric" });

/** A moment, in seconds: the time of day if it is today, with the date before it if not. */
export function formatWhen(at: number, now: number): string {
  const date = new Date(at * 1000);
  const today = new Date(now * 1000).toDateString() === date.toDateString();
  return today ? clock.format(date) : `${day.format(date)}, ${clock.format(date)}`;
}
