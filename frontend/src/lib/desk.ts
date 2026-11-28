"use client";

import { useCallback, useEffect, useReducer, useRef } from "react";

import type { DeskDetail, DeskEvent, DeskMessage, DeskStats, DeskTicket } from "./protocol";
import { socketUrl } from "./session";
import { agentName, deskUrl, rememberAgent, workspaceKey } from "./workspace";

/**
 * The service desk in the page. Like the conversation, it is a fold over what the server sends:
 * the tickets of the workspace when the desk connects, then every ticket that is opened or
 * changes, and the full case of a ticket when the desk asks for it.
 */
export interface DeskState {
  connection: "connecting" | "open" | "down";
  agent: string; // the name this desk acts under, as the server took it
  tickets: DeskTicket[]; // newest first
  stats: DeskStats | null;
  details: Record<string, DeskDetail>;
  /** The last thing the server turned down, with the ticket it was about. `n` counts them. */
  refusal: { key: string | null; message: string; n: number } | null;
}

export const initialDesk: DeskState = { connection: "connecting", agent: "", tickets: [], stats: null, details: {}, refusal: null };

export type DeskAction = { type: "event"; event: DeskEvent } | { type: "connection"; status: DeskState["connection"] } | { type: "dismiss" };

function withTicket(tickets: DeskTicket[], ticket: DeskTicket): DeskTicket[] {
  const others = tickets.filter((t) => t.key !== ticket.key);
  return [...others, ticket].sort((a, b) => b.created_at - a.created_at || a.id.localeCompare(b.id));
}

export function deskReduce(state: DeskState, action: DeskAction): DeskState {
  if (action.type === "connection") return { ...state, connection: action.status };
  if (action.type === "dismiss") return { ...state, refusal: null };
  const e = action.event;
  switch (e.type) {
    case "desk.ready":
      // A fresh list. Cases that were open are asked for again by whoever shows them.
      return { ...state, connection: "open", agent: e.agent, tickets: e.tickets, stats: e.stats, details: {} };
    case "desk.ticket": {
      const held = state.details[e.ticket.key];
      // The header of an open case follows the ticket at once. The rest of it is asked for again.
      const details = held ? { ...state.details, [e.ticket.key]: { ...held, ticket: e.ticket } } : state.details;
      return { ...state, tickets: withTicket(state.tickets, e.ticket), stats: e.stats, details };
    }
    case "desk.detail":
      return { ...state, tickets: withTicket(state.tickets, e.ticket), details: { ...state.details, [e.ticket.key]: e } };
    case "desk.error":
      return { ...state, refusal: { key: e.key, message: e.message, n: (state.refusal?.n ?? 0) + 1 } };
    default:
      return state;
  }
}

export interface Desk {
  state: DeskState;
  /** Ask for the full case of a ticket, and keep it fresh while it is the one on screen. */
  open(key: string | null): void;
  assign(key: string, assignee: string | null): void;
  setStatus(key: string, status: DeskTicket["status"]): void;
  setPriority(key: string, priority: DeskTicket["priority"]): void;
  note(key: string, body: string): void;
  reply(key: string, body: string, status?: DeskTicket["status"]): void;
  refund(key: string, invoiceId: string): void;
  /** Act under another name from now on. */
  rename(agent: string): void;
  dismiss(): void;
}

const BACKOFF_MS = [1_000, 2_000, 4_000, 8_000];

export function useDesk(): Desk {
  const [state, dispatch] = useReducer(deskReduce, initialDesk);
  const socket = useRef<WebSocket | null>(null);
  const shown = useRef<string | null>(null); // the ticket whose case is on screen
  const reconnect = useRef<() => void>(() => {});

  useEffect(() => {
    let stopped = false;
    let attempt = 0;
    let timer: number | undefined;

    const connect = () => {
      if (stopped) return;
      const ws = new WebSocket(deskUrl(socketUrl()));
      socket.current = ws;
      const current = () => !stopped && socket.current === ws;
      ws.onopen = () => {
        if (current()) ws.send(JSON.stringify({ type: "desk.hello", workspace: workspaceKey(), agent: agentName() } satisfies DeskMessage));
      };
      ws.onmessage = (message) => {
        if (!current() || typeof message.data !== "string") return;
        let event: DeskEvent;
        try {
          event = JSON.parse(message.data) as DeskEvent;
        } catch {
          return;
        }
        if (typeof event !== "object" || event === null) return;
        dispatch({ type: "event", event });
        if (event.type === "desk.ready") attempt = 0;
        // The case on screen is asked for again when the desk comes back, and whenever its
        // ticket changes: someone else replied, or the customer's conversation went on.
        const stale = event.type === "desk.ready" || (event.type === "desk.ticket" && event.ticket.key === shown.current);
        if (stale && shown.current) ws.send(JSON.stringify({ type: "desk.open", key: shown.current } satisfies DeskMessage));
      };
      ws.onclose = () => {
        if (!current()) return;
        socket.current = null;
        dispatch({ type: "connection", status: "down" });
        const wait = BACKOFF_MS[Math.min(attempt, BACKOFF_MS.length - 1)];
        attempt += 1;
        timer = window.setTimeout(connect, wait);
      };
    };
    reconnect.current = () => {
      window.clearTimeout(timer);
      const old = socket.current;
      socket.current = null;
      old?.close();
      attempt = 0;
      dispatch({ type: "connection", status: "connecting" });
      connect();
    };
    connect();
    return () => {
      stopped = true;
      window.clearTimeout(timer);
      socket.current?.close();
      socket.current = null;
    };
  }, []);

  const send = useCallback((message: DeskMessage) => {
    const ws = socket.current;
    if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify(message));
  }, []);

  const open = useCallback(
    (key: string | null) => {
      shown.current = key;
      if (key) send({ type: "desk.open", key });
    },
    [send],
  );
  const assign = useCallback((key: string, assignee: string | null) => send({ type: "desk.assign", key, assignee }), [send]);
  const setStatus = useCallback((key: string, status: DeskTicket["status"]) => send({ type: "desk.status", key, status }), [send]);
  const setPriority = useCallback((key: string, priority: DeskTicket["priority"]) => send({ type: "desk.priority", key, priority }), [send]);
  const note = useCallback((key: string, body: string) => send({ type: "desk.note", key, body }), [send]);
  const reply = useCallback(
    (key: string, body: string, status?: DeskTicket["status"]) => send({ type: "desk.reply", key, body, status: status ?? null }),
    [send],
  );
  const refund = useCallback((key: string, invoiceId: string) => send({ type: "desk.refund", key, invoice_id: invoiceId }), [send]);
  const rename = useCallback((agent: string) => {
    rememberAgent(agent);
    reconnect.current(); // the name is given with the hello, so the desk says hello again
  }, []);
  const dismiss = useCallback(() => dispatch({ type: "dismiss" }), []);

  return { state, open, assign, setStatus, setPriority, note, reply, refund, rename, dismiss };
}
