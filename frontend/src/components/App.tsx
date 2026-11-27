"use client";

import { useEffect, useRef, useState, useSyncExternalStore } from "react";

import { type DemoAccount, credentials } from "@/lib/demo";
import { useSession } from "@/lib/session";

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
  // While voice is on, the space bar stops the reply, wherever the keyboard is not typing or
  // pressing something else.
  useEffect(() => {
    if (!voiceOn) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.code !== "Space" || event.repeat || event.metaKey || event.ctrlKey || event.altKey) return;
      const target = event.target instanceof Element ? event.target : null;
      if (target?.closest("input, textarea, select, button, a, summary, [role='tab'], [contenteditable='true']")) return;
      event.preventDefault();
      speech.interrupt();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [voiceOn, speech]);

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
          <VoicePanel
            voice={voice}
            mic={speech.mic}
            out={speech.out}
            onInterrupt={speech.interrupt}
            onEnd={speech.stop}
            onDismiss={session.dismiss}
          />
          <Composer
            ready={online && !state.busy && state.pending === null}
            // With voice on, the voice panel has the one Stop there is: it stops the reply in the
            // ear as well as the turn behind it.
            busy={online && state.busy && !voiceOn}
            onSend={session.say}
            onStop={session.cancel}
            restored={state.draft}
            picked={picked}
            onAccount={pickAccount}
            voice={voice.offered ? { on: voiceOn, ready: online, onToggle: voiceOn ? speech.stop : speech.start } : null}
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
