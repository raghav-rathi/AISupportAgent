import { useEffect, useId, useRef, useState, useSyncExternalStore } from "react";

import type { Voice } from "@/lib/state";
import type { Levels } from "@/lib/voice";

import { Button, Caption, Glyph } from "./ui";

const ACTIVITY: Record<NonNullable<Voice["activity"]>, string> = {
  listening: "Listening",
  hearing: "Hearing you",
  thinking: "Thinking",
  speaking: "Speaking",
};

const BAR = 3; // CSS pixels
const GAP = 2;

const REDUCED_MOTION = "(prefers-reduced-motion: reduce)";

const DISCLOSURE =
  "You are talking to an AI assistant. Your speech goes to Deepgram, which turns it into text, and the replies are spoken by a " +
  "synthetic voice. This demo keeps no audio. It keeps the words, as the text of this conversation.";

function useReducedMotion(): boolean {
  return useSyncExternalStore(
    (onChange) => {
      const query = window.matchMedia(REDUCED_MOTION);
      query.addEventListener("change", onChange);
      return () => query.removeEventListener("change", onChange);
    },
    () => window.matchMedia(REDUCED_MOTION).matches,
    () => false,
  );
}

/**
 * The last second or two of one audio signal, as bars. It draws itself and re-renders nothing.
 * With reduced motion asked for, it is drawn once, flat, and only its colour says it is active.
 */
function Meter({ label, levels, active }: { label: string; levels: Levels; active: boolean }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const still = useReducedMotion();
  useEffect(() => {
    const element = canvas.current;
    if (!element) return;
    // Read once: the colour only changes with `active`, which runs this again.
    const color = getComputedStyle(element).color;
    let frame = 0;
    const draw = () => {
      if (!still) frame = requestAnimationFrame(draw);
      const context = element.getContext("2d");
      if (!context) return;
      const scale = window.devicePixelRatio || 1;
      const width = element.clientWidth;
      const height = element.clientHeight;
      // A canvas has whole pixels. Comparing unrounded sizes at a fractional ratio would
      // reallocate it on every frame.
      const pixelWidth = Math.round(width * scale);
      const pixelHeight = Math.round(height * scale);
      if (element.width !== pixelWidth || element.height !== pixelHeight) {
        element.width = pixelWidth;
        element.height = pixelHeight;
      }
      context.setTransform(scale, 0, 0, scale, 0, 0);
      context.clearRect(0, 0, width, height);
      context.fillStyle = color;
      const values = levels.read();
      const count = Math.min(values.length, Math.floor((width + GAP) / (BAR + GAP)));
      for (let i = 0; i < count; i++) {
        // Speech sits low on a linear scale. The root brings it up to where it can be seen.
        const level = still ? 0 : Math.sqrt(Math.min(1, values[values.length - count + i] * 4));
        const bar = Math.max(2, level * height);
        context.beginPath();
        context.roundRect(i * (BAR + GAP), (height - bar) / 2, BAR, bar, BAR / 2);
        context.fill();
      }
    };
    frame = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(frame);
  }, [levels, active, still]);
  return (
    <div className="flex flex-col gap-4">
      <Caption>{label}</Caption>
      <canvas ref={canvas} role="img" aria-label={`${label}: audio level`} className={`h-32 w-[138px] ${active ? "text-ivory-text" : "text-slate-border"}`} />
    </div>
  );
}

/** What voice involves, shown before the microphone opens for the first time in a page load. */
function Consent({ onStart, onDecline }: { onStart: () => void; onDecline: () => void }) {
  const start = useRef<HTMLButtonElement>(null);
  const described = useId();
  // The visitor pressed Voice to get here, so the choice it asks for takes the focus.
  useEffect(() => start.current?.focus(), []);
  return (
    <section aria-label="Voice" className="mx-auto w-full max-w-760 px-24 pb-12">
      <div className="flex flex-col gap-12 rounded-cards bg-graphite-card px-20 py-16">
        <p id={described} className="text-body-sm leading-[1.45]">
          {DISCLOSURE}
        </p>
        <div className="flex gap-8">
          {/* Only this opens the microphone. */}
          <Button ref={start} variant="surface" aria-describedby={described} onClick={onStart}>
            Start microphone
          </Button>
          <Button variant="quiet" onClick={onDecline}>
            Not now
          </Button>
        </div>
      </div>
    </section>
  );
}

/** What a screen reader hears about voice: that it came on, that it went off, and why. */
function announcement(voice: Voice): string {
  if (voice.status === "on") return "Voice is on.";
  if (voice.status === "starting") return "";
  return voice.notice ? `Voice is off: ${voice.notice}.` : "Voice is off.";
}

interface Props {
  voice: Voice;
  mic: Levels;
  out: Levels;
  /** Stop what is running: a typed turn, or the voice reply. Null when there is nothing to stop. */
  onStop: (() => void) | null;
  onEnd: () => void;
  onDismiss: () => void;
  /** The visitor pressed Voice for the first time in this page load, and is asked to confirm. */
  asking: boolean;
  onStart: () => void;
  onDecline: () => void;
}

/** Voice while it is on: who is making sound, what the assistant is doing, and how to stop it. */
export function VoicePanel({ voice, mic, out, onStop, onEnd, onDismiss, asking, onStart, onDecline }: Props) {
  // The one live region of voice. It changes when voice goes on or off, or with a new reason,
  // and not when a notice is dismissed or the activity changes: those would talk over the reply.
  const [said, setSaid] = useState({ status: voice.status, notice: voice.notice, text: "" });
  if (said.status !== voice.status || said.notice !== voice.notice) {
    const text = said.status !== voice.status || voice.notice ? announcement(voice) : said.text;
    setSaid({ status: voice.status, notice: voice.notice, text });
  }
  const region = (
    <p role="status" className="sr-only">
      {said.text}
    </p>
  );

  if (voice.status === "off") {
    return (
      <>
        {region}
        {asking ? (
          <Consent onStart={onStart} onDecline={onDecline} />
        ) : voice.notice ? (
          <div className="mx-auto w-full max-w-760 px-24 pb-12">
            <div className="flex items-center gap-12 rounded-cards bg-graphite-card px-20 py-12 text-body-sm leading-[1.45]">
              <Glyph name="alert" />
              <p className="flex-1">Voice is off: {voice.notice}. You can go on in text.</p>
              <Button variant="quiet" onClick={onDismiss} className="h-32 px-12">
                Dismiss
              </Button>
            </div>
          </div>
        ) : null}
      </>
    );
  }
  const starting = voice.status === "starting";
  return (
    <>
      {region}
      <section aria-label="Voice" className="mx-auto w-full max-w-760 px-24 pb-12">
        <div className="flex flex-col gap-12 rounded-cards bg-graphite-card px-20 py-16">
          <div className="flex flex-wrap items-end gap-x-20 gap-y-12">
            <Meter label="You" levels={mic} active={voice.activity === "hearing"} />
            <Meter label="Assistant" levels={out} active={voice.activity === "speaking"} />
            {/* Not a live region: it changes with every breath and would talk over the reply. */}
            <p className="min-w-[138px] flex-1 pb-4 text-body-sm font-w480">{starting ? "Starting voice…" : ACTIVITY[voice.activity ?? "listening"]}</p>
            <div className="flex gap-8">
              {/* Always here while voice is on, so the keyboard focus never falls off it. It is
                  marked disabled rather than disabled outright: a disabled button drops its focus. */}
              <Button
                variant="surface"
                aria-disabled={onStop === null}
                aria-keyshortcuts="Space Escape"
                onClick={onStop ?? undefined}
                className="aria-disabled:cursor-not-allowed aria-disabled:opacity-40"
              >
                <Glyph name="stop" /> Stop
              </Button>
              <Button variant="quiet" onClick={onEnd}>
                End voice
              </Button>
            </div>
          </div>
          {/* The visitor's words as the recogniser hears them. They become the message when the turn ends. */}
          <p aria-live="off" className="min-h-[1.45em] break-words text-body-sm leading-[1.45] text-ash-text">
            {voice.partial ? `“${voice.partial}”` : ""}
          </p>
          <p className="text-caption leading-[1.4] text-ash-text">
            {DISCLOSURE}{" "}
            {voice.bargeIn
              ? "Speak to interrupt a reply, or use Stop, the space bar or Escape."
              : "This browser does not cancel the echo of the reply, so speaking over a reply does not stop it. Use Stop, the space bar or Escape."}
          </p>
        </div>
      </section>
    </>
  );
}
