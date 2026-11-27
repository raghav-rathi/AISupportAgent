/**
 * Voice in the page. The microphone goes to the server as binary frames, the reply's audio comes
 * back the same way, and the server is told how much of it has been played. The server decides
 * the rest: when the visitor's turn is over, what is said, and when a reply is cut off.
 *
 * `VoiceLink` is the one place that moves the page's voice status between off, starting and on.
 */

import { type PlayheadCause, VoiceAudio, type VoiceEngine, schedule } from "./audio";
import type { ClientMessage, ServerEvent } from "./protocol";
import type { Action } from "./state";

const TICK_MS = 100; // how often the server hears where the playhead is, while audio plays
const STALE_MS = 160; // a meter nobody has fed for this long falls to zero
const HALF_CHUNK_MS = 40; // a loud chunk is timed at its end. The voice was in it, on average half way.
export const START_TIMEOUT_MS = 10_000; // how long the server has to answer voice.start
export const FLUSH_TIMEOUT_MS = 2_000; // how long audio stays muted after Stop if the server's flush never comes
const OPEN = 1; // WebSocket.OPEN

/** An ID for one message or one start of voice. The server answers under it. */
export function newId(): string {
  return typeof crypto.randomUUID === "function"
    ? crypto.randomUUID()
    : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 12)}`;
}

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

/** The part of a WebSocket that voice uses. */
export interface VoiceSocket {
  readonly readyState: number;
  send(data: string | ArrayBuffer): void;
}

/** What can be swapped for tests. Each defaults to the real thing. */
export interface VoiceLinkOptions {
  createAudio?: () => VoiceEngine;
  /** Why voice cannot work in this page, or null when it can. */
  problem?: () => string | null;
  newAttempt?: () => string;
  now?: () => number;
}

export class VoiceLink {
  readonly mic = new Levels();
  readonly out = new Levels();
  private readonly createAudio: () => VoiceEngine;
  private readonly problem: () => string | null;
  private readonly newAttempt: () => string;
  private readonly now: () => number;
  private audio: VoiceEngine | null = null;
  private attempt: string | null = null; // the ID of the voice.start in flight or answered
  private cancelStartTimeout: (() => void) | null = null;
  private live = false; // the server has said that voice is on
  private stopPressed = false; // the next local flush is the visitor's own Stop
  private muted = false; // audio of the stopped reply may still arrive: drop it until the server's flush
  private cancelUnmute: (() => void) | null = null;
  private tickedAt = 0;
  // For the time from the visitor's last word to the first sound of the reply, as heard here.
  private floor = 0.01;
  private loudAt = 0;
  private waiting: { turnId: string; since: number; playedMs: number } | null = null;
  private socket: VoiceSocket | null = null;

  constructor(
    private readonly dispatch: (action: Action) => void,
    options: VoiceLinkOptions = {},
  ) {
    this.createAudio = options.createAudio ?? (() => new VoiceAudio());
    this.problem = options.problem ?? VoiceAudio.problem;
    this.newAttempt = options.newAttempt ?? newId;
    this.now = options.now ?? (() => performance.now());
  }

  /** The connection to the server of the moment. Voice ends with the connection it began on. */
  use(socket: VoiceSocket | null): void {
    this.socket = socket;
  }

  private transmit(data: string | ArrayBuffer): boolean {
    if (this.socket?.readyState !== OPEN) return false;
    this.socket.send(data);
    return true;
  }

  private send(message: ClientMessage): boolean {
    return this.transmit(JSON.stringify(message));
  }

  /** Ask for the microphone and turn voice on. The browser may ask the visitor first. */
  async start(): Promise<void> {
    if (this.audio) return;
    const problem = this.problem();
    if (problem) {
      this.dispatch({ type: "voice", status: "off", notice: problem });
      return;
    }
    const audio = this.createAudio();
    this.audio = audio;
    this.dispatch({ type: "voice", status: "starting" });
    try {
      const info = await audio.start({
        onChunk: (pcm) => {
          if (this.live) this.transmit(pcm);
        },
        onMicLevel: (level) => this.hear(level),
        onOutLevel: (level) => this.out.push(level, this.now()),
        onPlayhead: (playedMs, bufferedMs, after) => this.report(playedMs, bufferedMs, after),
        onDrained: () => {},
        onMicLost: () => this.lose(audio, "the microphone was disconnected"),
        onSuspended: () => this.lose(audio, "voice stopped while the page was in the background"),
      });
      if (this.audio !== audio) return; // turned off while the browser was asking
      // Without echo cancellation the microphone hears the reply, and the server is told.
      const bargeIn = info.echoCancellation !== false;
      this.dispatch({ type: "voice", status: "starting", bargeIn });
      const attempt = this.newAttempt();
      this.attempt = attempt;
      if (!this.send({ type: "voice.start", echo_cancellation: bargeIn, attempt })) {
        this.end("the connection is down");
        return;
      }
      this.cancelStartTimeout = schedule(START_TIMEOUT_MS, () => {
        if (this.attempt !== attempt || this.live) return;
        this.send({ type: "voice.stop" });
        this.end("the server did not answer");
      });
    } catch (error) {
      if (this.audio === audio) this.end(microphoneProblem(error));
    }
  }

  /** Turn voice off from this side. The server hears it even when the audio is already gone. */
  stop(): void {
    this.send({ type: "voice.stop" });
    this.end(null);
  }

  /** The connection went away, and with it the server's side of voice. */
  dropped(): void {
    if (this.audio) this.end("the connection dropped");
  }

  /** Stop the reply that is playing or on its way. */
  interrupt(): void {
    if (!this.audio || !this.live || this.muted) return;
    this.stopPressed = true;
    // Until the server's flush says it has stopped sending, what arrives is the stopped reply.
    this.muted = true;
    this.cancelUnmute = schedule(FLUSH_TIMEOUT_MS, () => {
      this.muted = false;
    });
    this.audio.flush(); // the worklet answers with where the playhead stood
  }

  /** A binary frame from the server: a piece of the reply's audio. */
  play(pcm: ArrayBuffer): void {
    if (this.live && !this.muted) this.audio?.play(pcm);
  }

  /**
   * What the server says that concerns voice. True when the event is the server turning down this
   * page's start of voice: that is reported here, as the reason voice is off, and is no business
   * of the reducer's.
   */
  handle(event: ServerEvent): boolean {
    const audio = this.audio;
    if (!audio) return false;
    switch (event.type) {
      case "voice.started":
        if (!this.current(event.attempt)) break;
        this.live = true;
        this.cancelStartTimeout?.();
        this.dispatch({ type: "voice", status: "on" });
        break;
      case "voice.stopped":
        if (this.current(event.attempt)) this.end(event.reason === "client" ? null : event.message || "it was stopped");
        break;
      case "voice.playback":
        if (event.action === "pause") audio.pause();
        else if (event.action === "resume") audio.resume();
        else {
          // Every Stop is answered by a flush. After it, audio is the next reply's.
          this.muted = false;
          this.cancelUnmute?.();
          audio.flush();
        }
        break;
      case "error":
        if (!this.refuses(event)) break;
        this.end(event.message);
        return true;
      case "turn.started":
        if (this.live && event.mode === "voice" && event.turn_id) {
          this.waiting = { turnId: event.turn_id, since: this.loudAt, playedMs: audio.playedMs() };
        }
        break;
    }
    return false;
  }

  /** An event about the attempt under way. An empty attempt comes from a server that sends none. */
  private current(attempt: string): boolean {
    return this.attempt !== null && (attempt === "" || attempt === this.attempt);
  }

  /** An error that turns down the voice.start of the attempt under way. */
  private refuses(event: Extract<ServerEvent, { type: "error" }>): boolean {
    if (this.attempt === null || this.live) return false;
    if (event.client_id) return event.client_id === this.attempt;
    // A server that names no attempt still says which errors are about voice.
    return event.code === "voice_unavailable";
  }

  /** The audio of this attempt failed on its own. The server is told, and voice ends. */
  private lose(audio: VoiceEngine, notice: string): void {
    if (this.audio !== audio) return;
    if (this.attempt !== null) this.send({ type: "voice.stop" });
    this.end(notice);
  }

  private release(): void {
    const audio = this.audio;
    this.audio = null;
    this.attempt = null;
    this.live = false;
    this.stopPressed = false;
    this.muted = false;
    this.waiting = null;
    this.cancelStartTimeout?.();
    this.cancelUnmute?.();
    void audio?.stop();
  }

  private end(notice: string | null): void {
    this.release();
    this.dispatch({ type: "voice", status: "off", notice });
  }

  private hear(level: number): void {
    const now = this.now();
    this.mic.push(level, now);
    // The quietest the room gets, followed slowly. Sound well above it is the visitor speaking.
    this.floor = Math.min(level, this.floor * 1.01 + 0.00002);
    if (level > Math.max(0.015, this.floor * 4)) this.loudAt = now;
  }

  private report(playedMs: number, bufferedMs: number, after: PlayheadCause): void {
    const now = this.now();
    const waiting = this.waiting;
    if (waiting && playedMs > waiting.playedMs + 1) {
      // The reply has begun to play. Its first sample left the worklet this long ago, and reached
      // the loudspeaker the output latency later.
      const heardAt = now - (playedMs - waiting.playedMs) + (this.audio?.latencyMs() ?? 0);
      this.waiting = null;
      if (waiting.since > 0) {
        const ms = Math.max(0, heardAt - (waiting.since - HALF_CHUNK_MS));
        this.dispatch({ type: "voice-latency", turnId: waiting.turnId, ms });
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
