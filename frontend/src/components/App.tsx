"use client";

import { useEffect, useRef, useState, useSyncExternalStore } from "react";

import { type DemoAccount, credentials } from "@/lib/demo";
import { useSession } from "@/lib/session";
import { type KeyFocus, stopKey, stopTarget } from "@/lib/stop";

import { Composer, type Incoming } from "./Composer";
import { Conversation } from "./Conversation";
import { Drawer } from "./Drawer";
import { TopBar } from "./TopBar";
import { VoicePanel } from "./VoicePanel";
import { Button, Glyph } from "./ui";

/** True on a screen too narrow for the drawer to sit beside the conversation. */
const NARROW = "(max-width: 1023px)";

function useNarrow(): boolean {
  return useSyncExternalStore(
    (onChange) => {
      const query = window.matchMedia(NARROW);
      query.addEventListener("change", onChange);
      return () => query.removeEventListener("change", onChange);
    },
    () => window.matchMedia(NARROW).matches,
    () => false,
  );
}

/**
 * Where the keyboard was, for the Stop keys. `byPointer` is the element that got the focus from a
 * click or a tap, if the focus is still there. `:focus-visible` cannot answer that inside a key
 * handler: the browser starts showing the focus on the very key press that is being handled, so
 * a button the mouse left focused matched it, and the space bar pressed that button.
 */
function keyFocus(target: Element | null, byPointer: Element | null): KeyFocus {
  if (!target) return "none";
  if (target.closest("input, textarea, select, [contenteditable='true']")) return "field";
  const control = target.closest("button, a, summary, [role='tab'], [role='menuitem']");
  if (!control) return "none";
  return control instanceof HTMLButtonElement && control === byPointer ? "pointer-button" : "control";
}

export function App() {
  const session = useSession();
  const { state } = session;
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [inspected, setInspected] = useState<string | null>(null); // null follows the newest turn
  const [picked, setPicked] = useState<Incoming | null>(null);
  // Which simulated failures are queued, and at which point of the conversation. Kept here so
  // the mark survives closing the drawer or changing its tab.
  const [queued, setQueued] = useState<Record<string, number>>({});
  const drawerToggle = useRef<HTMLButtonElement>(null);
  // The element that holds the focus, if a click or a tap put it there. See keyFocus.
  const pointerFocus = useRef<Element | null>(null);
  useEffect(() => {
    let pointerAt = -Infinity;
    const onPointer = () => {
      pointerAt = performance.now();
    };
    const onFocus = (event: FocusEvent) => {
      // A click moves the focus within the same few milliseconds. Tab moves it without one.
      const clicked = performance.now() - pointerAt < 1000 && event.target instanceof Element;
      pointerFocus.current = clicked ? (event.target as Element).closest("button") : null;
    };
    window.addEventListener("pointerdown", onPointer, true);
    window.addEventListener("focusin", onFocus, true);
    return () => {
      window.removeEventListener("pointerdown", onPointer, true);
      window.removeEventListener("focusin", onFocus, true);
    };
  }, []);
  const narrow = useNarrow();

  const online = state.connection === "open";
  const down = state.connection === "reconnecting" || state.connection === "paused" || state.connection === "stopped";
  const startOver = () => {
    setInspected(null);
    setQueued({});
    session.startOver();
  };
  const closeDrawer = () => {
    setDrawerOpen(false);
    drawerToggle.current?.focus();
  };
  const pickAccount = (account: DemoAccount) => setPicked((before) => ({ text: credentials(account), n: (before?.n ?? 0) + 1 }));
  const overlay = drawerOpen && narrow; // the drawer covers the conversation
  const voice = state.voice;
  const speech = session.voice;
  const voiceOn = voice.status !== "off";
  // The panel's one Stop: a typed turn is cancelled as a turn, anything else of voice is
  // interrupted. Null when there is nothing to stop.
  const stopping = stopTarget(state);
  const stop = stopping === "turn" ? session.cancel : stopping === "reply" ? speech.interrupt : null;
  // While voice is on, the space bar and Escape press that Stop. See stopKey for when they do not.
  useEffect(() => {
    if (!voiceOn) return;
    const onKey = (event: KeyboardEvent) => {
      const target = event.target instanceof Element ? event.target : null;
      const decision = stopKey(
        {
          key: event.key,
          repeat: event.repeat,
          modified: event.metaKey || event.ctrlKey || event.altKey,
          defaultPrevented: event.defaultPrevented,
          focus: keyFocus(target, pointerFocus.current),
          fieldHasSelection:
            (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement) && target.selectionStart !== target.selectionEnd,
          dialogOpen: document.querySelector("[role='dialog'], [role='menu']") !== null,
        },
        stop !== null,
      );
      if (decision === "pass") return;
      event.preventDefault();
      if (decision === "stop") stop?.();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [voiceOn, stop]);

  return (
    <div className="flex h-dvh flex-col bg-onyx-canvas text-ivory-text">
      <TopBar
        state={state}
        drawerOpen={drawerOpen}
        drawerToggle={drawerToggle}
        onToggleDrawer={() => (drawerOpen ? closeDrawer() : setDrawerOpen(true))}
        onStartOver={startOver}
      />
      <div className="flex min-h-0 flex-1">
        <main className="flex min-w-0 flex-1 flex-col" inert={overlay}>
          <Conversation
            state={state}
            onSay={session.say}
            onAccount={pickAccount}
            onRetry={session.retry}
            onPassage={session.requestPassage}
            onInspect={(turnId) => {
              setInspected(turnId);
              setDrawerOpen(true);
            }}
          />
          {down || state.notice ? (
            <div className="mx-auto w-full max-w-760 px-24 pb-12">
              <div role="status" className="flex items-center gap-12 rounded-cards bg-graphite-card px-20 py-12 text-body-sm leading-[1.45]">
                <Glyph name="alert" />
                <p className="flex-1">
                  {state.connection === "stopped"
                    ? (state.closeReason ?? "The connection was closed.")
                    : state.connection === "paused"
                      ? "Paused after a while without activity. Your conversation is saved."
                      : state.connection === "reconnecting"
                        ? "The connection dropped. Reconnecting. A reply in progress keeps running and will show up here."
                        : `The server did not accept that: ${state.notice}.`}
                </p>
                {state.connection === "stopped" || state.connection === "paused" ? (
                  <Button variant="surface" onClick={session.reconnect} className="h-32 px-12">
                    {state.connection === "paused" ? "Reconnect" : "Try again"}
                  </Button>
                ) : !down ? (
                  <Button variant="quiet" onClick={session.dismiss} className="h-32 px-12">
                    Dismiss
                  </Button>
                ) : null}
              </div>
            </div>
          ) : null}
          <VoicePanel voice={voice} mic={speech.mic} out={speech.out} onStop={stop} onEnd={speech.stop} onDismiss={session.dismiss} />
          <Composer
            ready={online && !state.busy && state.pending === null}
            // With voice on, the voice panel has the one Stop there is. It stops a typed turn as
            // well as a spoken reply.
            busy={online && state.busy && !voiceOn}
            onSend={session.say}
            onStop={session.cancel}
            restored={state.draft}
            picked={picked}
            onAccount={pickAccount}
            voice={
              voice.offered
                ? { on: voiceOn, starting: voice.status === "starting", ready: online, onToggle: voiceOn ? speech.stop : speech.start }
                : null
            }
          />
        </main>
        {drawerOpen ? (
          <Drawer
            state={state}
            turnId={inspected}
            overlay={overlay}
            queued={queued}
            onQueued={(kind, at) => setQueued((before) => ({ ...before, [kind]: at }))}
            onSelectTurn={setInspected}
            onClose={closeDrawer}
            onInject={session.inject}
            onTraces={session.requestTraces}
            onPassage={session.requestPassage}
          />
        ) : null}
      </div>
    </div>
  );
}
