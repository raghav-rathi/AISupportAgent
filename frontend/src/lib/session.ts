"use client";

import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";

import type { ClientMessage, FaultInject, ServerEvent } from "./protocol";
import { planAfterClose } from "./reconnect";
import { type AppState, initialState, reduce } from "./state";
import { type Levels, VoiceLink, newId } from "./voice";
import { workspaceKey } from "./workspace";

const STORAGE_KEY = "fernhill.session";

/** The backend's WebSocket. Set at build time, or the same host on the backend's dev port. */
export function socketUrl(): string {
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

/** Unsent data past which the socket is behind. Microphone audio is dropped then, not queued. */
const MAX_BUFFERED = 65_536;

/** Send on a socket that is open. False when it is not, or when audio was dropped, and then nothing was sent. */
function transmit(ws: WebSocket | null, data: string | ArrayBuffer): boolean {
  if (ws?.readyState !== WebSocket.OPEN) return false;
  // Audio queued behind a slow connection reaches the server too late to be heard as speech, and
  // the queue would grow without end. A message is never dropped.
  if (typeof data !== "string" && ws.bufferedAmount > MAX_BUFFERED) return false;
  ws.send(data);
  return true;
}

export interface Session {
  state: AppState;
  /** Send a message. False when it could not be sent, and then nothing has changed. */
  say(text: string): boolean;
  retry(): void;
  cancel(): void;
  /** Queue a simulated failure. False when it could not be sent, and then nothing is queued. */
  inject(kind: FaultInject["kind"], tool?: string): boolean;
  startOver(): void;
  dismiss(): void;
  /** Connect again, when the connection is paused or stopped. */
  reconnect(): void;
  /** Ask for the prompt traces of one turn. They are stored on the server and not streamed. */
  requestTraces(turnId: string): void;
  /** Ask for the text of a passage that a search of this conversation returned. */
  requestPassage(chunkId: string): void;
  voice: {
    /** Turn the microphone on. The browser may ask the visitor first. */
    start(): void;
    stop(): void;
    /** Stop the reply that is being spoken, or about to be. */
    interrupt(): void;
    mic: Levels; // what the microphone picks up, for drawing
    out: Levels; // what is being played
  };
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
  // How to connect again while no connection is open or due. `auto` says whether the visitor
  // coming back to the tab is reason enough, or whether they have to ask.
  const wake = useRef<{ go: () => void; auto: boolean } | null>(null);
  // One link between the audio and the server for the life of the page. Each connection is
  // handed to it as it opens.
  const [voice] = useState(() => new VoiceLink(dispatch));
  // Passages asked for and not answered yet. A connection that drops loses the answer, so the
  // next one asks again.
  const looking = useRef(new Set<string>());

  const startOver = useCallback(() => {
    remember(null);
    cursor.current = { sessionId: null, lastSeq: 0 };
    looking.current.clear();
    dispatch({ type: "reset" });
    setGeneration((n) => n + 1);
  }, []);

  useEffect(() => {
    let stopped = false;
    let attempt = 0;
    let timer: number | undefined;
    if (cursor.current.sessionId === null) cursor.current.sessionId = stored();

    const open = () => {
      if (stopped) return;
      wake.current = null;
      const ws = new WebSocket(socketUrl());
      ws.binaryType = "arraybuffer";
      socket.current = ws;
      // Voice sends through `transmit` too, so its microphone audio gives way when the socket is behind.
      voice.use({
        get readyState() {
          return ws.readyState;
        },
        send: (data) => void transmit(ws, data),
      });
      // A socket that was replaced still delivers what was in flight. None of it may touch the
      // cursor, the stored ID or the state of the conversation that replaced it.
      const current = () => !stopped && socket.current === ws;

      ws.onopen = () => {
        if (!current()) return;
        const { sessionId, lastSeq } = cursor.current;
        // The workspace is what the service desk of this browser shows the tickets of.
        ws.send(JSON.stringify({ type: "hello", session_id: sessionId, last_seq: lastSeq, workspace: workspaceKey() }));
      };
      ws.onmessage = (message) => {
        if (!current()) return;
        if (typeof message.data !== "string") {
          voice.play(message.data as ArrayBuffer); // a binary frame is the reply's audio
          return;
        }
        let parsed: unknown;
        try {
          parsed = JSON.parse(message.data);
        } catch {
          return; // not ours to understand, and no reason to take the page down
        }
        // JSON that is not an object, such as null or a number, is no event either.
        if (typeof parsed !== "object" || parsed === null) return;
        const event = parsed as ServerEvent;
        let askAgain: string[] = [];
        if (event.type === "session.ready") {
          if (event.session_id !== cursor.current.sessionId) {
            cursor.current = { sessionId: event.session_id, lastSeq: 0 };
            looking.current.clear(); // another conversation, with passages of its own
          }
          askAgain = [...looking.current];
          remember(event.session_id);
        } else if (event.type === "session.caught_up") {
          // The connection works once it has caught up. One closed during the replay, for falling
          // behind, counts as a failed try, so the next one waits longer.
          attempt = 0;
        } else if (event.seq > cursor.current.lastSeq) {
          cursor.current.lastSeq = event.seq;
        }
        if (event.type === "passage.detail") looking.current.delete(event.chunk_id);
        if (event.type === "error" && event.code === "not_found") {
          if (event.chunk_id) looking.current.delete(event.chunk_id);
          else looking.current.clear();
        }
        // The server turning down this page's start of voice is reported by the voice link.
        if (voice.handle(event)) return;
        dispatch({ type: "event", event });
        for (const chunkId of askAgain) {
          ws.send(JSON.stringify({ type: "passage.get", chunk_id: chunkId }));
          dispatch({ type: "looked", chunkId });
        }
        if (event.type === "error" && event.code === "session_expired") startOver();
      };
      ws.onclose = (closed) => {
        if (!current()) return;
        socket.current = null;
        voice.dropped();
        voice.use(null);
        const plan = planAfterClose(closed.code, attempt);
        attempt += 1;
        if (plan.kind === "restart") {
          startOver();
        } else if (plan.kind === "stop") {
          dispatch({ type: "connection", status: "stopped", reason: plan.why });
          wake.current = { go: again, auto: false };
        } else if (plan.kind === "pause") {
          dispatch({ type: "connection", status: "paused" });
          wake.current = { go: again, auto: true };
        } else {
          dispatch({ type: "connection", status: "reconnecting", reason: closed.reason });
          timer = window.setTimeout(() => {
            // A tab nobody is looking at needs no connection, and a browser that knows it is
            // offline cannot make one. It connects when it is looked at and online.
            if (document.visibilityState === "hidden" || !navigator.onLine) wake.current = { go: open, auto: true };
            else open();
          }, plan.delayMs);
        }
      };
    };
    const again = () => {
      attempt = 0;
      dispatch({ type: "connection", status: "connecting" });
      open();
    };
    const onBack = () => {
      const waiting = wake.current;
      if (waiting?.auto && document.visibilityState === "visible" && navigator.onLine) {
        wake.current = null;
        waiting.go();
      }
    };
    const signs = ["visibilitychange", "focus", "pointerdown", "keydown", "online"] as const;
    for (const sign of signs) window.addEventListener(sign, onBack);
    open();

    return () => {
      stopped = true;
      wake.current = null;
      window.clearTimeout(timer);
      for (const sign of signs) window.removeEventListener(sign, onBack);
      voice.stop();
      voice.use(null);
      socket.current?.close();
      socket.current = null;
    };
  }, [generation, startOver, voice]);

  const send = useCallback((message: ClientMessage): boolean => transmit(socket.current, JSON.stringify(message)), []);

  const say = useCallback(
    (text: string): boolean => {
      const id = newId();
      if (!send({ type: "user.message", text, client_id: id })) return false;
      dispatch({ type: "sent", id, text });
      return true;
    },
    [send],
  );
  // A second click on Try again or Stop, before the page has caught up with the first, would be
  // turned down by the server and show up as an error for nothing.
  const lastAsk = useRef({ what: "", at: 0 });
  const once = useCallback(
    (message: ClientMessage) => {
      const now = Date.now();
      if (lastAsk.current.what === message.type && now - lastAsk.current.at < 1_000) return;
      lastAsk.current = { what: message.type, at: now };
      send(message);
    },
    [send],
  );
  const retry = useCallback(() => once({ type: "turn.retry" }), [once]);
  const cancel = useCallback(() => once({ type: "turn.cancel" }), [once]);
  const inject = useCallback(
    (kind: FaultInject["kind"], tool?: string): boolean => send({ type: "fault.inject", kind, tool, times: 1 }),
    [send],
  );
  const dismiss = useCallback(() => dispatch({ type: "dismiss" }), []);
  const reconnect = useCallback(() => {
    const waiting = wake.current;
    wake.current = null;
    waiting?.go();
  }, []);
  const requestTraces = useCallback(
    (turnId: string) => {
      if (send({ type: "trace.get", turn_id: turnId })) dispatch({ type: "asked", turnId });
    },
    [send],
  );
  const requestPassage = useCallback(
    (chunkId: string) => {
      if (!send({ type: "passage.get", chunk_id: chunkId })) return;
      looking.current.add(chunkId);
      dispatch({ type: "looked", chunkId });
    },
    [send],
  );

  const speech = useMemo(
    () => ({
      start: () => void voice.start(),
      stop: () => voice.stop(),
      interrupt: () => voice.interrupt(),
      mic: voice.mic,
      out: voice.out,
    }),
    [voice],
  );

  return { state, say, retry, cancel, inject, startOver, dismiss, reconnect, requestTraces, requestPassage, voice: speech };
}
