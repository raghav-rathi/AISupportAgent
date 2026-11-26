"use client";

import { useState } from "react";

import { useSession } from "@/lib/session";

import { Composer } from "./Composer";
import { Conversation } from "./Conversation";
import { Drawer } from "./Drawer";
import { TopBar } from "./TopBar";
import { Button, Glyph } from "./ui";

export function App() {
  const session = useSession();
  const { state } = session;
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [inspected, setInspected] = useState<string | null>(null); // null follows the newest turn

  const online = state.connection === "open";
  const startOver = () => {
    setInspected(null);
    session.startOver();
  };

  return (
    <div className="flex h-dvh flex-col bg-onyx-canvas text-ivory-text">
      <TopBar state={state} drawerOpen={drawerOpen} onToggleDrawer={() => setDrawerOpen((open) => !open)} onStartOver={startOver} />
      <div className="flex min-h-0 flex-1">
        <main className="flex min-w-0 flex-1 flex-col">
          <Conversation
            state={state}
            onPick={session.say}
            onRetry={session.retry}
            onInspect={(turnId) => {
              setInspected(turnId);
              setDrawerOpen(true);
            }}
          />
          {state.connection === "reconnecting" || state.notice ? (
            <div className="mx-auto w-full max-w-760 px-24 pb-12">
              <div role="status" className="flex items-center gap-12 rounded-cards bg-graphite-card px-20 py-12 text-body-sm leading-[1.45]">
                <Glyph name="alert" />
                <p className="flex-1">
                  {state.connection === "reconnecting"
                    ? "The connection dropped. Reconnecting. A reply in progress keeps running and will show up here."
                    : `The server did not accept that: ${state.notice}.`}
                </p>
                {state.connection !== "reconnecting" ? (
                  <Button variant="quiet" onClick={session.dismiss} className="h-32 px-12">
                    Dismiss
                  </Button>
                ) : null}
              </div>
            </div>
          ) : null}
          <Composer ready={online && !state.busy && state.pending === null} busy={online && state.busy} onSend={session.say} onStop={session.cancel} />
        </main>
        {drawerOpen ? (
          <Drawer state={state} turnId={inspected} onSelectTurn={setInspected} onClose={() => setDrawerOpen(false)} onInject={session.inject} />
        ) : null}
      </div>
    </div>
  );
}
