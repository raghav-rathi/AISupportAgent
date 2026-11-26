"use client";

import { useCallback, useEffect, useReducer, useRef, useState } from "react";

import type { ClientMessage, FaultInject, ServerEvent } from "./protocol";
import { type AppState, initialState, reduce } from "./state";

const STORAGE_KEY = "fernhill.session";
const MAX_WAIT_MS = 8_000;

/** The backend's WebSocket. Set at build time, or the same host on the backend's dev port. */
function socketUrl(): string {
  const configured = process.env.NEXT_PUBLIC_WS_URL;
  if (configured) return configured;
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${window.location.hostname}:8787/ws`;
}

function stored(): string | null {
  try {
    return window.localStorage.getItem(STORAGE_KEY);
  } catch {
    return null; // storage is blocked: the session lives as long as the page
  }
}

function remember(sessionId: string | null): void {
  try {
    if (sessionId) window.localStorage.setItem(STORAGE_KEY, sessionId);
    else window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    // see stored()
  }
}

export interface Session {
  state: AppState;
  say(text: string): void;
  retry(): void;
  cancel(): void;
  inject(kind: FaultInject["kind"], tool?: string): void;
  startOver(): void;
  dismiss(): void;
}

/**
 * One session over one WebSocket. The session ID is kept in the browser, so a reload or a dropped
 * connection picks the same conversation up: the client says the last event number it saw and
 * the server sends what it missed.
 */
export function useSession(): Session {
  const [state, dispatch] = useReducer(reduce, initialState);
  const [generation, setGeneration] = useState(0);
  const socket = useRef<WebSocket | null>(null);
  // Where this client is in which stream. Kept outside the state so a reconnect reads it at once.
  const cursor = useRef<{ sessionId: string | null; lastSeq: number }>({ sessionId: null, lastSeq: 0 });

  useEffect(() => {
    let stopped = false;
    let attempt = 0;
    let timer: number | undefined;
    if (cursor.current.sessionId === null) cursor.current.sessionId = stored();

    const open = () => {
      if (stopped) return;
      const ws = new WebSocket(socketUrl());
      socket.current = ws;
      ws.onopen = () => {
        const { sessionId, lastSeq } = cursor.current;
        ws.send(JSON.stringify({ type: "hello", session_id: sessionId, last_seq: lastSeq }));
      };
      ws.onmessage = (message) => {
        const event = JSON.parse(String(message.data)) as ServerEvent;
        if (event.type === "session.ready") {
          attempt = 0;
          if (event.session_id !== cursor.current.sessionId) {
            cursor.current = { sessionId: event.session_id, lastSeq: 0 };
          }
          remember(event.session_id);
        } else if (event.seq > cursor.current.lastSeq) {
          cursor.current.lastSeq = event.seq;
        }
        dispatch({ type: "event", event });
      };
      ws.onclose = (closed) => {
        if (stopped || socket.current !== ws) return;
        // The server gives a reason when it turns a visitor away, for example at its limit of
        // new conversations. A dropped network gives none.
        dispatch({ type: "connection", status: "reconnecting", reason: closed.reason });
        timer = window.setTimeout(open, Math.min(MAX_WAIT_MS, 500 * 2 ** attempt));
        attempt += 1;
      };
    };
    open();

    return () => {
      stopped = true;
      window.clearTimeout(timer);
      socket.current?.close();
      socket.current = null;
    };
  }, [generation]);

  const send = useCallback((message: ClientMessage): boolean => {
    const ws = socket.current;
    if (ws?.readyState !== WebSocket.OPEN) return false;
    ws.send(JSON.stringify(message));
    return true;
  }, []);

  const say = useCallback(
    (text: string) => {
      if (send({ type: "user.message", text })) dispatch({ type: "sent", text });
    },
    [send],
  );
  const retry = useCallback(() => void send({ type: "turn.retry" }), [send]);
  const cancel = useCallback(() => void send({ type: "turn.cancel" }), [send]);
  const inject = useCallback(
    (kind: FaultInject["kind"], tool?: string) => void send({ type: "fault.inject", kind, tool, times: 1 }),
    [send],
  );
  const startOver = useCallback(() => {
    remember(null);
    cursor.current = { sessionId: null, lastSeq: 0 };
    dispatch({ type: "reset" });
    setGeneration((n) => n + 1);
  }, []);
  const dismiss = useCallback(() => dispatch({ type: "dismiss" }), []);

  return { state, say, retry, cancel, inject, startOver, dismiss };
}
