import { useEffect, useRef, useState } from "react";

import { ACCOUNTS, type DemoAccount } from "@/lib/demo";

import { Button, Caption, Glyph } from "./ui";

const MAX_CHARS = 2_000; // the server's limit for one message
const MAX_HEIGHT = 168;

/** Text for the field that comes from outside it. `n` changes with every new piece. */
export interface Incoming {
  text: string;
  n: number;
}

interface Props {
  ready: boolean; // connected and no turn running
  busy: boolean; // a turn is running, so the action becomes Stop
  /** Returns false when the message could not be sent. The text then stays in the field. */
  onSend: (text: string) => boolean;
  onStop: () => void;
  restored: Incoming | null; // a message the server never took: it goes back into the field
  picked: Incoming | null; // credentials of a demo account, chosen here or elsewhere on the page
  onAccount: (account: DemoAccount) => void;
}

function AccountMenu({ onPick }: { onPick: (account: DemoAccount) => void }) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const close = (event: Event) => {
      if (event instanceof KeyboardEvent ? event.key === "Escape" : !root.current?.contains(event.target as Node)) setOpen(false);
    };
    window.addEventListener("keydown", close);
    window.addEventListener("pointerdown", close);
    return () => {
      window.removeEventListener("keydown", close);
      window.removeEventListener("pointerdown", close);
    };
  }, [open]);

  return (
    <div ref={root} className="relative">
      <button
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((was) => !was)}
        className="flex h-32 items-center gap-8 rounded-nav px-12 text-caption font-w480 text-ash-text hover:bg-obsidian-button hover:text-ivory-text"
      >
        <Glyph name="person" /> Demo accounts
      </button>
      {open ? (
        <div role="menu" aria-label="Demo accounts" className="absolute bottom-full left-0 z-10 mb-8 flex w-[min(420px,calc(100vw-48px))] flex-col gap-4 rounded-cards bg-graphite-card p-12">
          <Caption className="px-12 py-8">Puts the account number and PIN into your message</Caption>
          {ACCOUNTS.map((account) => (
            <button
              key={account.number}
              type="button"
              role="menuitem"
              onClick={() => {
                onPick(account);
                setOpen(false);
              }}
              className="flex flex-wrap items-baseline gap-x-12 gap-y-4 rounded-lg px-12 py-8 text-left text-body-sm leading-[1.45] hover:bg-obsidian-button"
            >
              <span className="font-w480">{account.name}</span>
              <span className="font-mono tabular-nums text-ash-text">
                {account.number} · PIN {account.pin}
              </span>
              <span className="basis-full text-caption leading-[1.4] text-ash-text">{account.note}</span>
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

export function Composer({ ready, busy, onSend, onStop, restored, picked, onAccount }: Props) {
  const [text, setText] = useState("");
  const field = useRef<HTMLTextAreaElement>(null);
  const sendable = ready && text.trim().length > 0;

  const fit = () => {
    const element = field.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, MAX_HEIGHT)}px`;
  };
  // Text from outside the field: a message the server never took, or the credentials of a demo
  // account. Each piece has a number, and a number not seen before is a new piece. It goes in
  // front of whatever has been typed since, while this component renders, so no effect has to
  // copy one state into another.
  const [taken, setTaken] = useState({ restored: restored?.n ?? 0, picked: 0 });
  const arrived = [
    restored && restored.n !== taken.restored ? restored.text : "",
    picked && picked.n !== taken.picked ? picked.text : "",
  ].filter(Boolean);
  if (arrived.length > 0) {
    setTaken({ restored: restored?.n ?? taken.restored, picked: picked?.n ?? taken.picked });
    setText((typed) => [...arrived, typed.trim()].filter(Boolean).join(" ").slice(0, MAX_CHARS));
  }
  const pieces = taken.restored + taken.picked;
  const mounted = useRef(false);
  useEffect(() => {
    // The field grows to fit what arrived, and the cursor goes there. Not on the first render:
    // nothing has arrived then, and focusing would open the keyboard on a phone.
    if (mounted.current) {
      fit();
      field.current?.focus();
    }
    mounted.current = true;
  }, [pieces]);

  const submit = () => {
    if (!sendable) return;
    // The field is cleared only once the message is on its way. A send that fails keeps the text.
    if (!onSend(text.trim())) return;
    setText("");
    if (field.current) field.current.style.height = "auto";
  };

  return (
    <div className="mx-auto w-full max-w-760 px-24 pb-[max(16px,env(safe-area-inset-bottom))]">
      <form
        onSubmit={(event) => {
          event.preventDefault();
          submit();
        }}
        className="flex items-end gap-8 rounded-inputs border border-mist-border/60 py-8 pl-20 pr-8 focus-within:border-mist-border"
      >
        <textarea
          ref={field}
          value={text}
          rows={1}
          maxLength={MAX_CHARS}
          placeholder="Ask a question"
          aria-label="Message"
          onChange={(event) => setText(event.target.value)}
          onInput={fit}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
              event.preventDefault();
              submit();
            }
          }}
          className="max-h-168 min-h-40 flex-1 resize-none bg-transparent py-8 text-body placeholder:text-ash-text focus:outline-none"
        />
        {busy ? (
          <Button variant="surface" onClick={onStop}>
            <Glyph name="stop" /> Stop
          </Button>
        ) : (
          <Button variant="primary" type="submit" disabled={!sendable}>
            Send <Glyph name="arrow" />
          </Button>
        )}
      </form>
      <div className="mt-8 flex flex-wrap items-center justify-between gap-x-16 gap-y-4">
        <AccountMenu onPick={onAccount} />
        <p className="text-caption font-w480 leading-[1.4] text-ash-text">A demo for an invented company. Do not enter real personal data.</p>
      </div>
    </div>
  );
}
