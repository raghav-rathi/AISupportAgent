import { useEffect, useRef } from "react";

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

/** The last second or two of one audio signal, as bars. It draws itself and re-renders nothing. */
function Meter({ label, levels, active }: { label: string; levels: Levels; active: boolean }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    let frame = 0;
    const draw = () => {
      frame = requestAnimationFrame(draw);
      const element = canvas.current;
      const context = element?.getContext("2d");
      if (!element || !context) return;
      const scale = window.devicePixelRatio || 1;
      const width = element.clientWidth;
      const height = element.clientHeight;
      if (element.width !== width * scale || element.height !== height * scale) {
        element.width = width * scale;
        element.height = height * scale;
      }
      context.setTransform(scale, 0, 0, scale, 0, 0);
      context.clearRect(0, 0, width, height);
      context.fillStyle = getComputedStyle(element).color;
      const values = levels.read();
      const count = Math.min(values.length, Math.floor((width + GAP) / (BAR + GAP)));
      for (let i = 0; i < count; i++) {
        // Speech sits low on a linear scale. The root brings it up to where it can be seen.
        const level = Math.sqrt(Math.min(1, values[values.length - count + i] * 4));
        const bar = Math.max(2, level * height);
        context.beginPath();
        context.roundRect(i * (BAR + GAP), (height - bar) / 2, BAR, bar, BAR / 2);
        context.fill();
      }
    };
    frame = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(frame);
  }, [levels]);
  return (
    <div className="flex flex-col gap-4">
      <Caption>{label}</Caption>
      <canvas ref={canvas} role="img" aria-label={`${label}: audio level`} className={`h-32 w-[138px] ${active ? "text-ivory-text" : "text-slate-border"}`} />
    </div>
  );
}

interface Props {
  voice: Voice;
  mic: Levels;
  out: Levels;
  onInterrupt: () => void;
  onEnd: () => void;
  onDismiss: () => void;
}

/** Voice while it is on: who is making sound, what the assistant is doing, and how to stop it. */
export function VoicePanel({ voice, mic, out, onInterrupt, onEnd, onDismiss }: Props) {
  if (voice.status === "off") {
    if (!voice.notice) return null;
    return (
      <div className="mx-auto w-full max-w-760 px-24 pb-12">
        <div role="status" className="flex items-center gap-12 rounded-cards bg-graphite-card px-20 py-12 text-body-sm leading-[1.45]">
          <Glyph name="alert" />
          <p className="flex-1">Voice is off: {voice.notice}. You can go on in text.</p>
          <Button variant="quiet" onClick={onDismiss} className="h-32 px-12">
            Dismiss
          </Button>
        </div>
      </div>
    );
  }
  const starting = voice.status === "starting";
  const canStop = voice.activity === "speaking" || voice.activity === "thinking";
  return (
    <section aria-label="Voice" className="mx-auto w-full max-w-760 px-24 pb-12">
      <div className="flex flex-col gap-12 rounded-cards bg-graphite-card px-20 py-16">
        <div className="flex flex-wrap items-end gap-x-20 gap-y-12">
          <Meter label="You" levels={mic} active={voice.activity === "hearing"} />
          <Meter label="Assistant" levels={out} active={voice.activity === "speaking"} />
          <p role="status" className="min-w-0 flex-1 pb-4 text-body-sm font-w480">
            {starting ? "Starting voice…" : ACTIVITY[voice.activity ?? "listening"]}
          </p>
          <div className="flex gap-8">
            {canStop ? (
              <Button variant="surface" onClick={onInterrupt}>
                <Glyph name="stop" /> Stop
              </Button>
            ) : null}
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
          You are talking to an AI assistant. Your speech goes to Deepgram, which turns it into text, and the replies are spoken by a
          synthetic voice. This demo keeps no audio. It keeps the words, as the text of this conversation.{" "}
          {voice.bargeIn
            ? "Speak to interrupt a reply, or use Stop or the space bar."
            : "This browser reports no echo cancellation, so speaking over a reply does not stop it. Use Stop or the space bar."}
        </p>
      </div>
    </section>
  );
}
