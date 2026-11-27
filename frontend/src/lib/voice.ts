/**
 * Voice in the page. The microphone goes to the server as binary frames, the reply's audio comes
 * back the same way, and the server is told how much of it has been played. The server decides
 * the rest: when the visitor's turn is over, what is said, and when a reply is cut off.
 */

import { type PlayheadCause, VoiceAudio } from "./audio";
import type { ClientMessage, ServerEvent } from "./protocol";
import type { Action } from "./state";

const TICK_MS = 100; // how often the server hears where the playhead is, while audio plays
const STALE_MS = 160; // a meter nobody has fed for this long falls to zero

/** The last few levels of one audio signal, for drawing. */
export class Levels {
  private readonly values = new Float32Array(28);
  private next = 0;
  private pushedAt = 0;

  push(level: number, now = performance.now()): void {
    this.values[this.next] = level;
    this.next = (this.next + 1) % this.values.length;
    this.pushedAt = now;
  }

  /** Oldest first. Silence scrolls in when nothing arrives. */
  read(now = performance.now()): Float32Array {
    if (now - this.pushedAt > STALE_MS) this.push(0, now);
    const ordered = new Float32Array(this.values.length);
    for (let i = 0; i < ordered.length; i++) ordered[i] = this.values[(this.next + i) % ordered.length];
    return ordered;
  }
}

function microphoneProblem(error: unknown): string {
  const name = error instanceof DOMException ? error.name : "";
  if (name === "NotAllowedError" || name === "SecurityError") return "the browser did not allow the microphone";
  if (name === "NotFoundError" || name === "OverconstrainedError") return "no microphone was found";
  return "the microphone could not be opened";
}

export class VoiceLink {
  readonly mic = new Levels();
  readonly out = new Levels();
  private audio: VoiceAudio | null = null;
  private live = false; // the server has said that voice is on
  private stopPressed = false; // the next flush is the visitor's own Stop
  private tickedAt = 0;
  // For the time from the visitor's last word to the first sound of the reply, as heard here.
  private floor = 0.01;
  private loudAt = 0;
  private waiting: { turnId: string; since: number; playedMs: number } | null = null;
  private socket: WebSocket | null = null;

  constructor(private readonly dispatch: (action: Action) => void) {}

  /** The connection to the server of the moment. Voice ends with the connection it began on. */
  use(socket: WebSocket | null): void {
    this.socket = socket;
  }

  private transmit(data: string | ArrayBuffer): boolean {
    if (this.socket?.readyState !== WebSocket.OPEN) return false;
    this.socket.send(data);
    return true;
  }

  private send(message: ClientMessage): boolean {
    return this.transmit(JSON.stringify(message));
  }

  /** Ask for the microphone and turn voice on. The browser may ask the visitor first. */
  async start(): Promise<void> {
    if (this.audio) return;
    if (!VoiceAudio.supported()) {
      this.dispatch({ type: "voice", status: "off", notice: "this browser cannot record audio" });
      return;
    }
    const audio = new VoiceAudio();
    this.audio = audio;
    this.dispatch({ type: "voice", status: "starting" });
    try {
      const info = await audio.start({
        onChunk: (pcm) => {
          if (this.live) this.transmit(pcm);
        },
        onMicLevel: (level) => this.hear(level),
        onOutLevel: (level) => this.out.push(level),
        onPlayhead: (playedMs, bufferedMs, after) => this.report(playedMs, bufferedMs, after),
        onDrained: () => {},
      });
      if (this.audio !== audio) return; // turned off while the browser was asking
      // Without echo cancellation the microphone hears the reply, and the server is told.
      const bargeIn = info.echoCancellation !== false;
      this.dispatch({ type: "voice", status: "starting", bargeIn });
      if (!this.send({ type: "voice.start", echo_cancellation: bargeIn })) this.end("the connection is down");
    } catch (error) {
      if (this.audio === audio) this.end(microphoneProblem(error));
    }
  }

  /** Turn voice off from this side. */
  stop(): void {
    if (!this.audio) return;
    this.send({ type: "voice.stop" });
    this.end(null);
  }

  /** The connection went away, and with it the server's side of voice. */
  dropped(): void {
    if (this.audio) this.end("the connection dropped");
  }

  /** Stop the reply that is playing or on its way. */
  interrupt(): void {
    if (!this.audio || !this.live) return;
    this.stopPressed = true;
    this.audio.flush(); // the worklet answers with where the playhead stood
  }

  /** A binary frame from the server: a piece of the reply's audio. */
  play(pcm: ArrayBuffer): void {
    if (this.live) this.audio?.play(pcm);
  }

  /** What the server says that concerns the audio. The page's state is the reducer's business. */
  handle(event: ServerEvent): void {
    const audio = this.audio;
    if (!audio) return;
    switch (event.type) {
      case "voice.started":
        this.live = true;
        break;
      case "voice.stopped":
        this.release();
        break;
      case "voice.playback":
        if (event.action === "pause") audio.pause();
        else if (event.action === "resume") audio.resume();
        else audio.flush();
        break;
      case "error":
        // Voice was asked for and the server said no. The page's state says why.
        if (!this.live) this.release();
        break;
      case "turn.started":
        if (this.live && event.mode === "voice" && event.turn_id) {
          this.waiting = { turnId: event.turn_id, since: this.loudAt, playedMs: audio.playedMs() };
        }
        break;
    }
  }

  private release(): void {
    const audio = this.audio;
    this.audio = null;
    this.live = false;
    this.stopPressed = false;
    this.waiting = null;
    void audio?.stop();
  }

  private end(notice: string | null): void {
    this.release();
    this.dispatch({ type: "voice", status: "off", notice });
  }

  private hear(level: number): void {
    const now = performance.now();
    this.mic.push(level, now);
    // The quietest the room gets, followed slowly. Sound well above it is the visitor speaking.
    this.floor = Math.min(level, this.floor * 1.01 + 0.00002);
    if (level > Math.max(0.015, this.floor * 4)) this.loudAt = now;
  }

  private report(playedMs: number, bufferedMs: number, after: PlayheadCause): void {
    const now = performance.now();
    const waiting = this.waiting;
    if (waiting && playedMs > waiting.playedMs + 1) {
      // The reply has begun to play. Its first sample left the loudspeaker this long ago.
      const began = now - (playedMs - waiting.playedMs);
      this.waiting = null;
      if (waiting.since > 0) {
        this.dispatch({ type: "voice-latency", turnId: waiting.turnId, ms: Math.max(0, began - waiting.since) });
      }
    }
    if (!this.live) return;
    if (after === "flush" && this.stopPressed) {
      this.stopPressed = false;
      this.send({ type: "voice.interrupt", played_ms: playedMs });
      return;
    }
    if (after === "tick" || after === "resume") {
      if (now - this.tickedAt < TICK_MS) return;
      this.tickedAt = now;
    }
    const cause = after === "resume" ? "tick" : after;
    this.send({ type: "voice.playhead", played_ms: playedMs, buffered_ms: bufferedMs, after: cause });
  }
}
