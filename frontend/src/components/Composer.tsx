import { useRef, useState } from "react";

import { Button, Glyph } from "./ui";

const MAX_CHARS = 2_000; // the server's limit for one message
const MAX_HEIGHT = 168;

interface Props {
  ready: boolean; // connected and no turn running
  busy: boolean; // a turn is running, so the action becomes Stop
  onSend: (text: string) => void;
  onStop: () => void;
}

export function Composer({ ready, busy, onSend, onStop }: Props) {
  const [text, setText] = useState("");
  const field = useRef<HTMLTextAreaElement>(null);
  const sendable = ready && text.trim().length > 0;

  const submit = () => {
    if (!sendable) return;
    onSend(text.trim());
    setText("");
    if (field.current) field.current.style.height = "auto";
  };

  return (
    <div className="mx-auto w-full max-w-760 px-24 pb-16">
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
          onInput={(event) => {
            const element = event.currentTarget;
            element.style.height = "auto";
            element.style.height = `${Math.min(element.scrollHeight, MAX_HEIGHT)}px`;
          }}
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
      <p className="mt-12 text-center text-caption font-w480 leading-[1.4] text-ash-text">
        A demo for an invented company. Do not enter real personal data.
      </p>
    </div>
  );
}
