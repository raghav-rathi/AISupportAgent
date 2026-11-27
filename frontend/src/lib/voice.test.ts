import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { PlayheadCause, VoiceAudioEvents, VoiceAudioInfo, VoiceEngine } from "./audio";
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

  get on(): VoiceAudioEvents {
    if (!this.events) throw new Error("not started");
    return this.events;
  }

  start(events: VoiceAudioEvents): Promise<VoiceAudioInfo> {
    this.events = events;
    return Promise.resolve({ echoCancellation: true, contextRate: 48000 });
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
  flush(): void {
    this.calls.push("flush");
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

function setup(problem: string | null = null) {
  const actions: Action[] = [];
  const audios: FakeAudio[] = [];
  const socket = new FakeSocket();
  const clock = { now: 1000 };
  let attempts = 0;
  const link = new VoiceLink((action) => actions.push(action), {
    createAudio: () => {
      const audio = new FakeAudio();
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
const playback = (action: "pause" | "resume" | "flush") => event({ type: "voice.playback", action });
const refusal = (clientId: string | null, code = "voice_unavailable") =>
  event({ type: "error", code, message: "voice is not available", client_id: clientId });
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
});

describe("Stop", () => {
  it("drops the stopped reply's audio until the server's flush, then plays again", async () => {
    const v = await live();
    v.link.play(frame());
    expect(v.audio.played).toHaveLength(1);
    v.link.interrupt();
    expect(v.audio.calls).toEqual(["flush"]);
    v.audio.on.onPlayhead(1200, 0, "flush"); // the worklet answers the flush with its playhead
    expect(v.socket.messages().at(-1)).toEqual({ type: "voice.interrupt", played_ms: 1200 });
    v.link.play(frame()); // sent by the server before it heard of the Stop
    expect(v.audio.played).toHaveLength(1);
    v.link.handle(playback("flush"));
    v.link.play(frame()); // the next reply
    expect(v.audio.played).toHaveLength(2);
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
    expect(v.audio.calls).toEqual(["flush"]);
  });
});

describe("the playhead", () => {
  it("is reported at most every 100 ms while it ticks, and a report after resume counts as a tick", async () => {
    const v = await live();
    const report = (playedMs: number, after: PlayheadCause) => v.audio.on.onPlayhead(playedMs, 500, after);
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
      { type: "voice.playhead", played_ms: 100, buffered_ms: 500, after: "tick" },
      { type: "voice.playhead", played_ms: 200, buffered_ms: 500, after: "tick" },
      { type: "voice.playhead", played_ms: 220, buffered_ms: 500, after: "pause" },
      { type: "voice.playhead", played_ms: 220, buffered_ms: 500, after: "tick" },
    ]);
  });

  it("times voice to voice from half way into the last loud chunk to the sound leaving the speaker", async () => {
    const v = await live();
    v.audio.latency = 30;
    v.audio.on.onMicLevel(0.5); // loud, at 1000
    v.advance(500);
    v.link.handle(event({ type: "turn.started", turn_id: "t1", mode: "voice", text: "Hi", client_id: null }));
    v.advance(700); // now 2200
    v.audio.on.onPlayhead(50, 400, "tick"); // the first sample left the worklet at 2150
    expect(v.actions).toContainEqual({ type: "voice-latency", turnId: "t1", ms: 2150 + 30 - (1000 - 40) });
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
