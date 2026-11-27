import type { Ref } from "react";

import { PLAN_LABELS } from "@/lib/labels";
import type { AppState, Connection } from "@/lib/state";

import { Button, Glyph, Tag } from "./ui";

const CONNECTION: Record<Connection, string> = {
  connecting: "Connecting",
  open: "Connected",
  reconnecting: "Reconnecting",
  paused: "Paused",
  stopped: "Not connected",
};

interface Props {
  state: AppState;
  drawerOpen: boolean;
  drawerToggle: Ref<HTMLButtonElement>; // focus returns here when the drawer closes
  onToggleDrawer: () => void;
  onStartOver: () => void;
  inert: boolean; // the drawer covers the page, and only it takes focus and clicks
}

export function TopBar({ state, drawerOpen, drawerToggle, onToggleDrawer, onStartOver, inert }: Props) {
  const online = state.connection === "open";
  const customer = state.customer;
  return (
    <header inert={inert} className="flex h-72 shrink-0 items-center gap-16 bg-onyx-canvas/80 px-24 backdrop-blur-[20px] max-sm:gap-8 max-sm:px-16">
      <div className="flex items-center gap-12">
        <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden="true">
          <circle cx="12" cy="12" r="10" />
          <circle cx="12" cy="12" r="5.5" />
          <circle cx="12" cy="12" r="1.2" fill="currentColor" />
        </svg>
        <p className="whitespace-nowrap font-arcadiadisplay text-body-lg font-w480 tracking-[0.02em]">
          Fernhill <span className="text-ash-text max-sm:hidden">Support</span>
        </p>
      </div>

      <div className="flex min-w-0 flex-1 items-center justify-end gap-8">
        <p className="mr-8 flex items-center gap-8 text-caption font-w480 text-ash-text max-sm:mr-0" role="status">
          <span className={`size-6 rounded-full ${online ? "bg-ivory-text" : "animate-pulse bg-slate-border"}`} />
          <span className="max-sm:sr-only">{CONNECTION[state.connection]}</span>
        </p>
        {customer ? (
          <Tag className="min-w-0">
            <Glyph name="check" className="mr-4" />
            <span className="sr-only">Verified: </span>
            <span className="truncate">
              {customer.contact_name.split(" ")[0]}
              <span className="max-sm:hidden"> {customer.contact_name.split(" ").slice(1).join(" ")}</span> · {PLAN_LABELS[customer.tier]}
            </span>
          </Tag>
        ) : null}
        <Button variant="quiet" onClick={onStartOver} aria-label="New conversation" className="max-sm:w-44 max-sm:px-0">
          <Glyph name="plus" />
          <span className="max-sm:hidden">New conversation</span>
        </Button>
        <Button
          ref={drawerToggle}
          onClick={onToggleDrawer}
          aria-label="Observability"
          aria-expanded={drawerOpen}
          aria-controls={drawerOpen ? "observability" : undefined}
          className="max-sm:w-44 max-sm:px-0"
        >
          <Glyph name="panel" />
          <span className="max-sm:hidden">Observability</span>
        </Button>
      </div>
    </header>
  );
}
