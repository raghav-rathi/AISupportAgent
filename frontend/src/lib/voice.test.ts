import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { type PlayheadCause, SoundBlockedError, type VoiceAudioEvents, type VoiceAudioInfo, type VoiceEngine } from "./audio";
import type { ClientMessage, ServerEvent } from "./protocol";
import type { Action } from "./state";
import { FLUSH_TIMEOUT_MS, START_TIMEOUT_MS, VoiceLink, type VoiceSocket } from "./voice";

/** Audio that records what it was asked to do, and lets the test speak for the worklets. */
class FakeAudio implements VoiceEngine {
  played: ArrayBuffer[] = [];
  calls: string[] = [];
  latency = 0;
  playhead = 0;
  private events: VoiceAudioEvents | null = null;

  constructor(
    private readonly echo: boolean | string | null = true,
    private readonly failure: unknown = null,
  ) {}

  get on(): VoiceAudioEvents {
    if (!this.events) throw new Error("not started");
    return this.events;
  }

  start(events: VoiceAudioEvents): Promise<VoiceAudioInfo> {
    this.events = events;
    if (this.failure) return Promise.reject(this.failure);
    return Promise.resolve({ echoCancellation: this.echo, contextRate: 48000 });
  }
  play(pcm: ArrayBuffer): void {
    this.played.push(pcm);
  }
  pause(): void {
    this.calls.push("pause");
  }
  resume(): void {
    this.calls.push("resume");
  }
  flush(epoch?: number): void {
    this.calls.push(epoch === undefined ? "flush" : `flush ${epoch}`);
  }
  rates(input: number, output: number): void {
    this.calls.push(`rates ${input} ${output}`);
  }
  playedMs(): number {
    return this.playhead;
  }
  latencyMs(): number {
    return this.latency;
  }
  stop(): Promise<void> {
    this.calls.push("stop");
    return Promise.resolve();
  }
}

class FakeSocket implements VoiceSocket {
  readyState = 1;
  sent: (string | ArrayBuffer)[] = [];
  send(data: string | ArrayBuffer): void {
    this.sent.push(data);
  }
  messages(): ClientMessage[] {
    return this.sent.flatMap((data) => (typeof data === "string" ? [JSON.parse(data) as ClientMessage] : []));
  }
}

function setup(problem: string | null = null, echo: boolean | string | null = true, failure: unknown = null) {
  const actions: Action[] = [];
  const audios: FakeAudio[] = [];
  const socket = new FakeSocket();
  const clock = { now: 1000 };
  let attempts = 0;
  const link = new VoiceLink((action) => actions.push(action), {
    createAudio: () => {
      const audio = new FakeAudio(echo, failure);
      audios.push(audio);
      return audio;
    },
    problem: () => problem,
    newAttempt: () => `a${++attempts}`,
    now: () => clock.now,
  });
  link.use(socket);
  return {
    link,
    actions,
    audios,
    socket,
    get audio(): FakeAudio {
      const audio = audios.at(-1);
      if (!audio) throw new Error("no audio");
      return audio;
    },
    advance(ms: number) {
      clock.now += ms;
    },
    /** What the page's voice status was told, in order. */
    statuses() {
      return actions.flatMap((action) => (action.type === "voice" ? [action.status] : []));
    },
    lastVoice() {
      return actions.filter((action) => action.type === "voice").at(-1);
    },
    latencies() {
      return actions.filter((action) => action.type === "voice-latency");
    },
  };
}

/** Voice that the server has turned on, under attempt a1. */
async function live() {
  const v = setup();
  await v.link.start();
  v.link.handle(started("a1"));
  return v;
}

const event = (fields: Record<string, unknown>) => ({ seq: 0, ts: 0, turn_id: null, ...fields }) as ServerEvent;
const started = (attempt: string) => event({ type: "voice.started", input_rate: 16000, output_rate: 24000, attempt });
const stopped = (attempt: string, reason = "idle", message = "voice turned off after a while of silence") =>
  event({ type: "voice.stopped", reason, message, attempt });
const playback = (action: "pause" | "resume" | "flush", id: number) => event({ type: "voice.playback", action, id });
const refusal = (clientId: string | null, code = "voice_unavailable") =>
  event({ type: "error", code, message: "voice is not available", client_id: clientId });
const turnStarted = (turnId: string, mode: "voice" | "text" = "voice") =>
  event({ type: "turn.started", turn_id: turnId, mode, text: "Hi", client_id: null });
const turnEnded = (type: "turn.finished" | "turn.failed" | "turn.interrupted", turnId: string) => event({ type, turn_id: turnId });
const frame = () => new ArrayBuffer(4800);

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("starting voice", () => {
  it("sends its attempt and turns on when the server answers that attempt", async () => {
    const v = setup();
    await v.link.start();
    expect(v.socket.messages()).toEqual([{ type: "voice.start", echo_cancellation: true, attempt: "a1" }]);
    expect(v.statuses()).toEqual(["starting", "starting"]);
    v.link.handle(started("a1"));
    expect(v.statuses().at(-1)).toBe("on");
  });

  it("says why when the page cannot do voice at all", async () => {
    const v = setup("voice needs HTTPS or localhost");
    await v.link.start();
    expect(v.actions).toEqual([{ type: "voice", status: "off", notice: "voice needs HTTPS or localhost" }]);
    expect(v.audios).toHaveLength(0);
  });

  it("gives up when the server answers neither yes nor no", async () => {
    const v = setup();
    await v.link.start();
    vi.advanceTimersByTime(START_TIMEOUT_MS - 1);
    expect(v.statuses().at(-1)).toBe("starting");
    vi.advanceTimersByTime(1);
    expect(v.lastVoice()).toEqual({ type: "voice", status: "off", notice: "the server did not answer" });
    expect(v.socket.messages().at(-1)).toEqual({ type: "voice.stop" });
    expect(v.audio.calls).toContain("stop");
  });

  it("does not time out once the server has answered", async () => {
    const v = await live();
    vi.advanceTimersByTime(START_TIMEOUT_MS * 2);
    expect(v.statuses().at(-1)).toBe("on");
  });

  it("offers barge-in only for echo cancellation that also removes what the page plays", async () => {
    const cases: [boolean | string | null, boolean][] = [
      ["all", true],
      [true, true],
      ["remote-only", false],
      [false, false],
      ["none", false],
      ["", false],
      [null, false],
    ];
    for (const [echo, bargeIn] of cases) {
      const v = setup(null, echo);
      await v.link.start();
      expect(v.socket.messages()).toEqual([{ type: "voice.start", echo_cancellation: bargeIn, attempt: "a1" }]);
    }
  });

  it("says that the browser kept the sound off when the audio could not start running", async () => {
    const v = setup(null, true, new SoundBlockedError());
    await v.link.start();
    expect(v.lastVoice()).toEqual({ type: "voice", status: "off", notice: "the browser did not allow sound to play" });
  });

  it("hands the rates the server names to the audio", async () => {
    const v = await live();
    expect(v.audio.calls).toContain("rates 16000 24000");
  });
});

describe("which events belong to this voice", () => {
  it("ignores the end of an earlier voice that arrives during a new start", async () => {
    const v = await live();
    v.link.stop();
    await v.link.start(); // attempt a2
    v.link.handle(stopped("a1", "client", ""));
    expect(v.audios[1].calls).not.toContain("stop");
    expect(v.statuses().at(-1)).toBe("starting");
    v.link.handle(started("a2"));
    expect(v.statuses().at(-1)).toBe("on");
    v.link.handle(stopped("a2"));
    expect(v.lastVoice()).toEqual({ type: "voice", status: "off", notice: "voice turned off after a while of silence" });
  });

  it("ignores an error that is about something else, and ends on the refusal of its start", async () => {
    const v = setup();
    await v.link.start();
    expect(v.link.handle(refusal(null, "not_found"))).toBe(false); // a passage that was not there
    expect(v.link.handle(refusal("m-1", "too_long"))).toBe(false); // a typed message
    expect(v.statuses().at(-1)).toBe("starting");
    expect(v.audio.calls).not.toContain("stop");
    expect(v.link.handle(refusal("a1", "budget_exhausted"))).toBe(true);
    expect(v.lastVoice()).toEqual({ type: "voice", status: "off", notice: "voice is not available" });
    expect(v.audio.calls).toContain("stop");
  });

  it("takes an event without an attempt as its own, as an older server sends them", async () => {
    const v = setup();
    await v.link.start();
    v.link.handle(started(""));
    expect(v.statuses().at(-1)).toBe("on");
    v.link.handle(stopped("", "client", ""));
    expect(v.lastVoice()).toEqual({ type: "voice", status: "off", notice: null });
  });

  it("turns off a voice the server started for an attempt that was given up", async () => {
    const v = setup();
    await v.link.start();
    v.link.stop();
    const before = v.actions.length;
    v.link.handle(started("a1"));
    expect(v.socket.messages().filter((message) => message.type === "voice.stop")).toHaveLength(2);
    expect(v.actions).toHaveLength(before); // the page stays off
  });

  it("leaves the stopping to the server when a newer start is on its way", async () => {
    const v = setup();
    await v.link.start();
    v.link.stop();
    await v.link.start(); // attempt a2 replaces a1 on the server
    v.link.handle(started("a1"));
    expect(v.socket.messages().filter((message) => message.type === "voice.stop")).toHaveLength(1);
    v.link.handle(started("a2"));
    expect(v.statuses().at(-1)).toBe("on");
  });
});

describe("Stop", () => {
  it("drops the stopped reply's audio until a flush numbered above the one it sent", async () => {
    const v = await live();
    v.link.handle(playback("flush", 1));
    v.link.play(frame());
    expect(v.audio.played).toHaveLength(1);
    v.link.interrupt();
    expect(v.audio.calls.at(-1)).toBe("flush");
    v.audio.on.onPlayhead(1200, 0, "stop", 1); // the worklet answers the Stop with its playhead
    expect(v.socket.messages().at(-1)).toEqual({ type: "voice.interrupt", played_ms: 1200, epoch: 1 });
    v.link.play(frame()); // sent by the server before it heard of the Stop
    v.link.handle(playback("flush", 1)); // the same flush again changes nothing
    v.link.play(frame());
    expect(v.audio.played).toHaveLength(1);
    v.link.handle(playback("flush", 2));
    v.link.play(frame()); // the next reply
    expect(v.audio.played).toHaveLength(2);
  });

  it("stays silent through a flush the server sent before it heard of the Stop", async () => {
    const v = await live();
    v.link.interrupt();
    v.link.handle(playback("flush", 1)); // crossed the Stop on the way
    v.link.play(frame());
    v.audio.on.onPlayhead(800, 0, "stop", 0);
    v.audio.on.onPlayhead(800, 0, "flush", 1);
    expect(v.socket.messages().slice(-2)).toEqual([
      { type: "voice.interrupt", played_ms: 800, epoch: 1 },
      { type: "voice.playhead", played_ms: 800, buffered_ms: 0, after: "flush", epoch: 1 },
    ]);
    v.link.play(frame());
    expect(v.audio.played).toHaveLength(0);
    v.link.handle(playback("flush", 2));
    v.link.play(frame());
    expect(v.audio.played).toHaveLength(1);
  });

  it("plays again after two seconds when the server's flush does not come", async () => {
    const v = await live();
    v.link.interrupt();
    vi.advanceTimersByTime(FLUSH_TIMEOUT_MS - 1);
    v.link.play(frame());
    expect(v.audio.played).toHaveLength(0);
    vi.advanceTimersByTime(1);
    v.link.play(frame());
    expect(v.audio.played).toHaveLength(1);
  });

  it("does nothing more when pressed again before the server's flush", async () => {
    const v = await live();
    v.link.interrupt();
    v.link.interrupt();
    expect(v.audio.calls.filter((call) => call === "flush")).toHaveLength(1);
  });
});

describe("pause and resume", () => {
  it("follows them when they come after the flush they name", async () => {
    const v = await live();
    v.link.handle(playback("pause", 0));
    v.link.handle(playback("resume", 0));
    expect(v.audio.calls.slice(-2)).toEqual(["pause", "resume"]);
  });

  it("keeps one that overtook its flush until that flush, and drops one from before the last flush", async () => {
    const v = await live();
    v.link.handle(playback("pause", 1));
    expect(v.audio.calls).not.toContain("pause");
    v.link.handle(playback("flush", 1));
    expect(v.audio.calls.slice(-2)).toEqual(["flush 1", "pause"]);
    v.link.handle(playback("flush", 2));
    v.link.handle(playback("resume", 1));
    expect(v.audio.calls.at(-1)).toBe("flush 2");
  });
});

describe("the playhead", () => {
  it("is reported at most every 100 ms while it ticks, and a report after resume counts as a tick", async () => {
    const v = await live();
    const report = (playedMs: number, after: PlayheadCause) => v.audio.on.onPlayhead(playedMs, 500, after, 0);
    report(100, "tick");
    v.advance(50);
    report(150, "tick"); // too soon
    v.advance(50);
    report(200, "tick");
    v.advance(20);
    report(220, "pause"); // a command is answered at once
    v.advance(100);
    report(220, "resume");
    const heads = v.socket.messages().filter((message) => message.type === "voice.playhead");
    expect(heads).toEqual([
      { type: "voice.playhead", played_ms: 100, buffered_ms: 500, after: "tick", epoch: 0 },
      { type: "voice.playhead", played_ms: 200, buffered_ms: 500, after: "tick", epoch: 0 },
      { type: "voice.playhead", played_ms: 220, buffered_ms: 500, after: "pause", epoch: 0 },
      { type: "voice.playhead", played_ms: 220, buffered_ms: 500, after: "tick", epoch: 0 },
    ]);
  });

  it("answers a flush with the number of that flush", async () => {
    const v = await live();
    v.link.handle(playback("flush", 1));
    v.link.handle(playback("flush", 2));
    v.audio.on.onPlayhead(300, 0, "flush", 1);
    v.audio.on.onPlayhead(300, 0, "flush", 2);
    expect(v.audio.calls.slice(-2)).toEqual(["flush 1", "flush 2"]);
    expect(v.socket.messages().slice(-2)).toEqual([
      { type: "voice.playhead", played_ms: 300, buffered_ms: 0, after: "flush", epoch: 1 },
      { type: "voice.playhead", played_ms: 300, buffered_ms: 0, after: "flush", epoch: 2 },
    ]);
  });
});

describe("voice to voice, as heard here", () => {
  it("times from half way into the last loud chunk to the sound leaving the speaker, once the turn has finished", async () => {
    const v = await live();
    v.audio.latency = 30;
    v.audio.on.onMicLevel(0.5); // loud, at 1000
    v.advance(500);
    v.link.handle(turnStarted("t1"));
    v.advance(700); // now 2200
    v.audio.on.onPlayhead(50, 400, "tick", 0); // the first sample left the worklet at 2150
    expect(v.latencies()).toEqual([]);
    v.link.handle(turnEnded("turn.finished", "t1"));
    expect(v.latencies()).toEqual([{ type: "voice-latency", turnId: "t1", ms: 2150 + 30 - (1000 - 40) }]);
  });

  it("times each turn from loud sound of its own", async () => {
    const v = await live();
    v.audio.on.onMicLevel(0.5);
    v.link.handle(turnStarted("t1"));
    v.link.handle(turnEnded("turn.finished", "t1"));
    v.audio.playhead = 1000;
    v.advance(60_000);
    v.audio.on.onMicLevel(0.01); // too quiet to count
    v.link.handle(turnStarted("t2"));
    v.link.handle(turnEnded("turn.finished", "t2"));
    v.audio.on.onPlayhead(1050, 0, "tick", 0);
    expect(v.latencies()).toEqual([]);
  });

  it("is not thrown off by a level that is not a number", async () => {
    const v = await live();
    v.audio.on.onMicLevel(Number.NaN);
    v.audio.on.onMicLevel(0.5);
    v.link.handle(turnStarted("t1"));
    v.link.handle(turnEnded("turn.finished", "t1"));
    v.audio.on.onPlayhead(50, 0, "tick", 0);
    expect(v.latencies()).toHaveLength(1);
  });

  it("gives no figure to a turn that failed, was cut off, or was typed", async () => {
    for (const end of ["turn.failed", "turn.interrupted"] as const) {
      const v = await live();
      v.audio.on.onMicLevel(0.5);
      v.link.handle(turnStarted("t1"));
      v.audio.on.onPlayhead(50, 0, "tick", 0);
      v.link.handle(turnEnded(end, "t1"));
      v.link.handle(turnEnded("turn.finished", "t1"));
      expect(v.latencies()).toEqual([]);
    }
    const v = await live();
    v.audio.on.onMicLevel(0.5);
    v.link.handle(turnStarted("t1"));
    v.link.handle(turnStarted("t2", "text"));
    v.audio.on.onPlayhead(50, 0, "tick", 0);
    v.link.handle(turnEnded("turn.finished", "t1"));
    expect(v.latencies()).toEqual([]);
  });
});

describe("ending voice", () => {
  it("tells the server even when no audio is left", () => {
    const v = setup();
    v.link.stop();
    expect(v.socket.messages()).toEqual([{ type: "voice.stop" }]);
    expect(v.actions).toEqual([{ type: "voice", status: "off", notice: null }]);
  });

  it("ends when the microphone goes away", async () => {
    const v = await live();
    v.audio.on.onMicLost();
    expect(v.socket.messages().at(-1)).toEqual({ type: "voice.stop" });
    expect(v.lastVoice()).toEqual({ type: "voice", status: "off", notice: "the microphone was disconnected" });
    expect(v.audio.calls).toContain("stop");
    // The audio that went is not voice any more: what it says later changes nothing.
    const before = v.actions.length;
    v.audio.on.onMicLost();
    expect(v.actions).toHaveLength(before);
  });

  it("ends when the browser suspended the audio for good", async () => {
    const v = await live();
    v.audio.on.onSuspended();
    expect(v.lastVoice()).toEqual({ type: "voice", status: "off", notice: "voice stopped while the page was in the background" });
  });
});
