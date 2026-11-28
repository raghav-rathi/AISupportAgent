import type { Ref } from "react";

import { PLAN_LABELS } from "@/lib/labels";
import type { AppState, Connection } from "@/lib/state";

import { Button, Glyph, type GlyphName, Tag } from "./ui";

const CONNECTION: Record<Connection, string> = {
  connecting: "Connecting",
  open: "Connected",
  reconnecting: "Reconnecting",
  paused: "Paused",
  stopped: "Not connected",
};

/** The two sides of the page: the customer's conversation, and the desk that gets its tickets. */
export type View = "assistant" | "desk";

const VIEWS: { view: View; label: string; glyph: GlyphName }[] = [
  { view: "assistant", label: "Assistant", glyph: "chat" },
  { view: "desk", label: "Desk", glyph: "inbox" },
];

interface Props {
  state: AppState;
  view: View;
  onView: (view: View) => void;
  waiting: number; // tickets at the desk that nobody has resolved
  deskOnline: boolean;
  drawerOpen: boolean;
  drawerToggle: Ref<HTMLButtonElement>; // focus returns here when the drawer closes
  onToggleDrawer: () => void;
  onStartOver: () => void;
  inert: boolean; // the drawer covers the page, and only it takes focus and clicks
}

export function TopBar({ state, view, onView, waiting, deskOnline, drawerOpen, drawerToggle, onToggleDrawer, onStartOver, inert }: Props) {
  const assistant = view === "assistant";
  const online = assistant ? state.connection === "open" : deskOnline;
  const customer = state.customer;
  return (
    <header inert={inert} className="flex h-64 shrink-0 items-center gap-16 bg-paper-white px-24 max-sm:gap-8 max-sm:px-16">
      <div className="flex items-center gap-8 text-obsidian">
        <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
          <circle cx="12" cy="12" r="10" />
          <circle cx="12" cy="12" r="5.5" />
          <circle cx="12" cy="12" r="1.2" fill="currentColor" />
        </svg>
        <p className="whitespace-nowrap text-body-sm font-semibold tracking-[-0.01em] max-sm:sr-only">Fernhill</p>
      </div>
      <span className="text-ash max-sm:hidden" aria-hidden="true">
        /
      </span>
      <nav aria-label="Views" className="flex items-center gap-4">
        {VIEWS.map((item) => (
          <button
            key={item.view}
            type="button"
            aria-current={view === item.view ? "page" : undefined}
            onClick={() => onView(item.view)}
            className={`inline-flex h-28 items-center gap-8 whitespace-nowrap rounded-md px-8 text-body-sm transition-colors ${
              view === item.view ? "bg-hairline text-obsidian" : "text-stone hover:text-obsidian"
            }`}
          >
            <Glyph name={item.glyph} className="max-sm:hidden" />
            {item.label}
            {item.view === "desk" && waiting > 0 ? (
              <span className="font-mono text-eyebrow tabular-nums text-obsidian" aria-label={`${waiting} unresolved`}>
                {waiting}
              </span>
            ) : null}
          </button>
        ))}
      </nav>

      <div className="flex min-w-0 flex-1 items-center justify-end gap-8">
        {/* On a phone the bar has no room to say that all is well. It still says when it is not. */}
        <p className={`label-mono mr-8 flex items-center gap-8 text-stone max-sm:mr-0 ${online ? "max-sm:sr-only" : ""}`} role="status">
          <span className={`size-6 rounded-full ${online ? "bg-terminal-green" : "animate-pulse bg-ash"}`} />
          <span className="max-md:sr-only">{assistant ? CONNECTION[state.connection] : deskOnline ? "Live" : "Connecting"}</span>
        </p>
        {assistant && customer ? (
          <Tag className="min-w-0 max-sm:max-w-[104px]">
            <Glyph name="check" className="mr-4 text-terminal-green" />
            <span className="sr-only">Verified: </span>
            <span className="truncate">
              {customer.contact_name.split(" ")[0]}
              <span className="max-sm:sr-only"> · {PLAN_LABELS[customer.tier]}</span>
            </span>
          </Tag>
        ) : null}
        {assistant ? (
          <>
            <Button variant="quiet" size="sm" onClick={onStartOver} aria-label="New conversation" className="max-sm:h-36 max-sm:w-36 max-sm:px-0">
              <Glyph name="plus" />
              <span className="max-sm:hidden">New conversation</span>
            </Button>
            <Button
              ref={drawerToggle}
              size="sm"
              onClick={onToggleDrawer}
              aria-label="Observability"
              aria-expanded={drawerOpen}
              aria-controls={drawerOpen ? "observability" : undefined}
              className="max-sm:h-36 max-sm:w-36 max-sm:px-0"
            >
              <Glyph name="panel" />
              <span className="max-sm:hidden">Observability</span>
            </Button>
          </>
        ) : null}
      </div>
    </header>
  );
}
