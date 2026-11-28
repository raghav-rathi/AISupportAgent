import { type ReactNode, useEffect, useRef } from "react";

import { ACCOUNTS, type DemoAccount } from "@/lib/demo";
import { type AppState, shownStatus } from "@/lib/state";

import { CaseUpdate } from "./CaseUpdate";
import { TurnView } from "./TurnView";
import { Caption, Card, Eyebrow, Glyph, Spinner } from "./ui";

/** Openers that show one behaviour each. The accounts are invented and public. */
const STARTERS = [
  { label: "Ask about plans", shows: "Answer with sources", text: "What plans do you offer, and what do they cost?" },
  {
    label: "Check the policy, then refund",
    shows: "Four tools in a row",
    text: "My account number is 10482917 and my PIN is 4721. What is the refund window on my plan? If my latest invoice is inside it, please refund it. It was billed by mistake.",
  },
  {
    label: "Refund above the limit",
    shows: "Ticket for the service desk",
    text: "My account number is 20573846 and my PIN is 9034. Please refund invoice INV-30388.",
  },
  { label: "Ask something off the map", shows: "No answer, so a hand-off", text: "Can I connect Fernhill to Microsoft Teams?" },
  { label: "Ask for a person", shows: "Ticket for the service desk", text: "I'd like to talk to a person about a contract question." },
];

function Welcome({ onSay, onAccount, ready }: { onSay: (text: string) => void; onAccount: (account: DemoAccount) => void; ready: boolean }) {
  return (
    <div className="mx-auto flex w-full max-w-640 flex-col gap-24 pb-40 pt-40 max-sm:pt-24">
      <div className="flex flex-col gap-16">
        <Eyebrow>Fernhill support</Eyebrow>
        <h1 className="text-heading-lg font-w450 text-carbon max-sm:text-heading">How can we help?</h1>
        <p className="max-w-520 text-body text-charcoal">
          Ask about plans, billing and refunds. The assistant checks who you are before it changes anything on an account. What it
          cannot settle goes to a person at the service desk, with a record of what it tried.
        </p>
      </div>
      <Card className="flex flex-col overflow-hidden">
        <Caption className="px-16 pb-8 pt-16">Try one</Caption>
        <ul className="flex flex-col">
          {STARTERS.map((starter) => (
            <li key={starter.label} className="border-t border-hairline">
              <button
                type="button"
                disabled={!ready}
                onClick={() => onSay(starter.text)}
                className="group flex w-full items-center gap-12 px-16 py-12 text-left hover:bg-paper-white disabled:cursor-not-allowed disabled:opacity-40"
              >
                <span className="flex-1 text-body-sm font-medium text-obsidian">{starter.label}</span>
                <span className="label-mono text-stone max-sm:hidden">{starter.shows}</span>
                <Glyph name="right" className="text-smoke group-hover:text-obsidian" />
              </button>
            </li>
          ))}
        </ul>
      </Card>
      <Card className="flex flex-col overflow-hidden">
        <Caption className="px-16 pb-8 pt-16">Demo accounts</Caption>
        <ul className="flex flex-col">
          {ACCOUNTS.map((account) => (
            <li key={account.number} className="border-t border-hairline">
              <button
                type="button"
                onClick={() => onAccount(account)}
                className="flex w-full flex-wrap items-baseline gap-x-12 gap-y-4 px-16 py-8 text-left text-caption hover:bg-paper-white"
              >
                <span className="font-mono tabular-nums text-obsidian">{account.number}</span>
                <span className="font-mono tabular-nums text-stone">PIN {account.pin}</span>
                <span className="text-charcoal">{account.note}</span>
              </button>
            </li>
          ))}
        </ul>
        <p className="border-t border-hairline px-16 py-12 text-caption text-stone">
          Choosing one puts its number and PIN into your message. The list stays under the message field. Fernhill is an invented
          company, and every conversation gets its own copy of this data, so a refund here changes nothing for anyone else.
        </p>
      </Card>
    </div>
  );
}

interface Props {
  state: AppState;
  onSay: (text: string) => void;
  onAccount: (account: DemoAccount) => void;
  onRetry: () => void;
  onInspect: (turnId: string) => void;
  onPassage: (chunkId: string) => void;
  onDesk: (ticketId: string) => void;
}

export function Conversation({ state, onSay, onAccount, onRetry, onInspect, onPassage, onDesk }: Props) {
  const scroller = useRef<HTMLDivElement>(null);
  const pinned = useRef(true); // the reader is at the bottom, so new text may pull the view down
  const turnCount = state.turns.length;
  const sending = state.pending !== null;

  // A message of the reader's own always brings the view back down, wherever they had scrolled.
  useEffect(() => {
    pinned.current = true;
  }, [turnCount, sending]);
  const ready = state.connection === "open";
  const empty = turnCount === 0 && !sending;
  // The welcome is read from its top. A conversation is read from its end.
  useEffect(() => {
    const element = scroller.current;
    if (element && pinned.current && !empty) element.scrollTop = element.scrollHeight;
  });
  // A new conversation starts at the top of the welcome, wherever the last one was left.
  useEffect(() => {
    if (empty && scroller.current) scroller.current.scrollTop = 0;
  }, [empty]);


  // Turns and what the service desk said, in the order they happened. A reply from a person
  // stands after the turn it followed and before the one that came later.
  const thread: { at: number; node: ReactNode }[] = state.turns.map((turn, index) => ({
    at: turn.startedTs,
    node: (
      <TurnView
        key={turn.id}
        turn={turn}
        status={shownStatus(state, turn)}
        canRetry={ready && state.resumable && !state.busy && index === turnCount - 1}
        onRetry={onRetry}
        onInspect={() => onInspect(turn.id)}
        passages={state.passages}
        onPassage={onPassage}
        voiceOn={state.voice.status === "on"}
        updates={state.updates}
        onDesk={onDesk}
      />
    ),
  }));
  for (const update of state.updates) thread.push({ at: update.ts, node: <CaseUpdate key={`case-${update.seq}`} update={update} /> });
  thread.sort((a, b) => a.at - b.at);

  return (
    <div
      ref={scroller}
      onScroll={(event) => {
        const element = event.currentTarget;
        pinned.current = element.scrollHeight - element.scrollTop - element.clientHeight < 96;
      }}
      className="relative min-h-0 flex-1 overflow-y-auto"
    >
      {empty ? (
        <div className="px-24 max-sm:px-16">
          <Welcome onSay={onSay} onAccount={onAccount} ready={ready && !state.busy} />
        </div>
      ) : (
        <div className="mx-auto flex w-full max-w-760 flex-col gap-32 px-24 pb-32 pt-32 max-sm:px-16">
          {thread.map((item) => item.node)}
          {state.pending !== null ? (
            <div className="flex flex-col gap-20">
              <div className="flex justify-end">
                <p className="max-w-[85%] whitespace-pre-wrap break-words rounded-md bg-hairline px-16 py-8 text-obsidian opacity-60">
                  {state.pending.text}
                </p>
              </div>
              <p className="flex items-center gap-8 font-mono text-caption text-stone" role="status">
                <Spinner /> {ready ? "Sending…" : "Waiting for the connection to see whether this arrived…"}
              </p>
            </div>
          ) : null}
        </div>
      )}
    </div>
  );
}
