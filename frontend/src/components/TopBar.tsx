import type { AppState } from "@/lib/state";

import { Button, Glyph, Tag } from "./ui";

const PLAN = { free: "Free", pro: "Pro", enterprise: "Enterprise" };

interface Props {
  state: AppState;
  drawerOpen: boolean;
  onToggleDrawer: () => void;
  onStartOver: () => void;
}

export function TopBar({ state, drawerOpen, onToggleDrawer, onStartOver }: Props) {
  const online = state.connection === "open";
  const customer = state.customer;
  return (
    <header className="flex h-72 shrink-0 items-center gap-16 bg-onyx-canvas/80 px-24 backdrop-blur-[20px]">
      <div className="flex items-center gap-12">
        <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden="true">
          <circle cx="12" cy="12" r="10" />
          <circle cx="12" cy="12" r="5.5" />
          <circle cx="12" cy="12" r="1.2" fill="currentColor" />
        </svg>
        <p className="whitespace-nowrap font-arcadiadisplay text-body-lg font-w480 tracking-[0.02em]">
          Fernhill <span className="text-ash-text">Support</span>
        </p>
      </div>

      <div className="flex flex-1 items-center justify-end gap-8">
        <p className="mr-8 flex items-center gap-8 text-caption font-w480 text-ash-text" role="status">
          <span className={`size-6 rounded-full ${online ? "bg-ivory-text" : "animate-pulse bg-slate-border"}`} />
          <span className="max-sm:sr-only">
            {online ? "Connected" : state.connection === "connecting" ? "Connecting" : "Reconnecting"}
          </span>
        </p>
        {customer ? (
          <Tag className="max-sm:hidden">
            <Glyph name="check" className="mr-4" />
            {customer.contact_name} · {PLAN[customer.tier]}
          </Tag>
        ) : null}
        <Button variant="quiet" onClick={onStartOver} aria-label="New conversation" className="max-sm:px-12">
          <Glyph name="plus" />
          <span className="max-sm:hidden">New conversation</span>
        </Button>
        <Button
          onClick={onToggleDrawer}
          aria-label="Observability"
          aria-expanded={drawerOpen}
          aria-controls="observability"
          className="max-sm:px-12"
        >
          <Glyph name="panel" />
          <span className="max-sm:hidden">Observability</span>
        </Button>
      </div>
    </header>
  );
}
