import { useEffect, useRef } from "react";

import { ACCOUNTS, type DemoAccount } from "@/lib/demo";
import { type AppState, shownStatus } from "@/lib/state";

import { TurnView } from "./TurnView";
import { Button, Caption, Spinner } from "./ui";

/** Openers that show one behaviour each. The accounts are invented and public. */
const STARTERS = [
  { label: "Ask about plans", text: "What plans do you offer, and what do they cost?" },
  {
    label: "Check the policy, then refund",
    text: "My account number is 10482917 and my PIN is 4721. What is the refund window on my plan? If my latest invoice is inside it, please refund it. It was billed by mistake.",
  },
  { label: "Refund above the limit", text: "My account number is 20573846 and my PIN is 9034. Please refund invoice INV-30388." },
  { label: "Ask something off the map", text: "Can I connect Fernhill to Microsoft Teams?" },
  { label: "Ask for a person", text: "I'd like to talk to a person about a contract question." },
];

function Welcome({ onSay, onAccount, ready }: { onSay: (text: string) => void; onAccount: (account: DemoAccount) => void; ready: boolean }) {
  return (
    <div className="mx-auto flex w-full max-w-640 flex-col items-center gap-32 pb-40 pt-72 text-center max-sm:pt-40">
      <div className="flex flex-col gap-16">
        <h1 className="font-arcadiadisplay text-heading-lg font-w480 max-sm:text-heading">How can we help?</h1>
        <p className="text-body-lg font-w420 text-ash-text">
          Ask about plans, billing and refunds. Before it changes anything on an account, the assistant checks who you are.
        </p>
      </div>
      <div className="flex flex-wrap justify-center gap-12">
        {STARTERS.map((starter) => (
          <Button key={starter.label} disabled={!ready} onClick={() => onSay(starter.text)}>
            {starter.label}
          </Button>
        ))}
      </div>
      <div className="flex w-full flex-col gap-16 rounded-cards bg-graphite-card p-32 text-left max-sm:p-24">
        <Caption>Demo accounts</Caption>
        <ul className="flex flex-col gap-4">
          {ACCOUNTS.map((account) => (
            <li key={account.number}>
              <button
                type="button"
                onClick={() => onAccount(account)}
                className="-mx-12 flex w-[calc(100%+24px)] flex-wrap items-baseline gap-x-12 gap-y-4 rounded-lg px-12 py-8 text-left text-body-sm leading-[1.45] hover:bg-obsidian-button"
              >
                <span className="font-mono tabular-nums">{account.number}</span>
                <span className="font-mono tabular-nums text-ash-text">PIN {account.pin}</span>
                <span className="text-ash-text">{account.note}</span>
              </button>
            </li>
          ))}
        </ul>
        <p className="text-caption leading-[1.4] text-ash-text">
          Choosing one puts its number and PIN into your message. The list stays under the message field. Fernhill is an invented
          company, and every conversation gets its own copy of this data, so a refund here changes nothing for anyone else.
        </p>
      </div>
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
}

export function Conversation({ state, onSay, onAccount, onRetry, onInspect, onPassage }: Props) {
  const scroller = useRef<HTMLDivElement>(null);
  const pinned = useRef(true); // the reader is at the bottom, so new text may pull the view down
  const turnCount = state.turns.length;
  const sending = state.pending !== null;

  // A message of the reader's own always brings the view back down, wherever they had scrolled.
  useEffect(() => {
    pinned.current = true;
  }, [turnCount, sending]);
  useEffect(() => {
    const element = scroller.current;
    if (element && pinned.current) element.scrollTop = element.scrollHeight;
  });

  const ready = state.connection === "open";
  const empty = turnCount === 0 && !sending;
  return (
    <div
      ref={scroller}
      onScroll={(event) => {
        const element = event.currentTarget;
        pinned.current = element.scrollHeight - element.scrollTop - element.clientHeight < 96;
      }}
      className="min-h-0 flex-1 overflow-y-auto"
    >
      {empty ? (
        <div className="px-24">
          <Welcome onSay={onSay} onAccount={onAccount} ready={ready && !state.busy} />
        </div>
      ) : (
        <div className="mx-auto flex w-full max-w-760 flex-col gap-40 px-24 pb-32 pt-40">
          {state.turns.map((turn, index) => (
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
            />
          ))}
          {state.pending !== null ? (
            <div className="flex flex-col gap-20">
              <div className="flex justify-end">
                <p className="max-w-[85%] whitespace-pre-wrap break-words rounded-cards bg-obsidian-button px-20 py-12 opacity-70">
                  {state.pending.text}
                </p>
              </div>
              <p className="flex items-center gap-12 text-body-sm text-ash-text" role="status">
                <Spinner /> {ready ? "Sending…" : "Waiting for the connection to see whether this arrived…"}
              </p>
            </div>
          ) : null}
        </div>
      )}
    </div>
  );
}
