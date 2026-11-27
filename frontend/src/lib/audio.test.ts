import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { CaptureProcessor, Downsampler } from "../../public/worklets/capture.js";
import { PlaybackProcessor } from "../../public/worklets/playback.js";
import { type PageVisibility, type SuspendableContext, keepRunning } from "./audio";

type Message = { type: string; [key: string]: unknown };
type ChunkMessage = { type: "chunk"; pcm: ArrayBuffer; level: number };
type PlayheadMessage = { type: "playhead"; playedMs: number; bufferedMs: number; level: number };
type DrainedMessage = { type: "drained"; playedMs: number };

interface TestPort {
  postMessage: (message: Message, transfer?: Transferable[]) => void;
  onmessage: ((event: { data: unknown }) => void) | null;
}

const BLOCK = 128;

/** Replaces the port so the test sees every posted message. */
function listen(processor: object): { port: TestPort; posted: Message[]; transfers: Transferable[][] } {
  // The port type is inferred from plain JavaScript, so the test states the shape it uses.
  const target = processor as { port: TestPort };
  const port = target.port;
  const posted: Message[] = [];
  const transfers: Transferable[][] = [];
  port.postMessage = (message, transfer) => {
    posted.push(message);
    transfers.push(transfer ?? []);
  };
  return { port, posted, transfers };
}

function send(port: TestPort, data: unknown): void {
  port.onmessage?.({ data });
}

function sine(frequency: number, rate: number, length: number, amplitude = 1): Float32Array {
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) out[i] = amplitude * Math.sin((2 * Math.PI * frequency * i) / rate);
  return out;
}

function rms(samples: Float32Array): number {
  let sum = 0;
  for (const s of samples) sum += s * s;
  return Math.sqrt(sum / samples.length);
}

function db(ratio: number): number {
  return 20 * Math.log10(ratio);
}

/** Deterministic noise so block and whole-call results can be compared. */
function noise(length: number): Float32Array {
  const out = new Float32Array(length);
  let state = 12345;
  for (let i = 0; i < length; i++) {
    state = (state * 1103515245 + 12345) % 2147483648;
    out[i] = state / 1073741824 - 1;
  }
  return out;
}

function concat(parts: Float32Array[]): Float32Array {
  const out = new Float32Array(parts.reduce((total, part) => total + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.length;
  }
  return out;
}

function inBlocks(downsampler: Downsampler, input: Float32Array): Float32Array {
  const parts: Float32Array[] = [];
  for (let i = 0; i < input.length; i += BLOCK) parts.push(downsampler.push(input.subarray(i, i + BLOCK)));
  return concat(parts);
}

function pcm16(samples: ArrayLike<number>): ArrayBuffer {
  const out = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) out[i] = Math.round(samples[i] * 32767);
  return out.buffer;
}

function constant(value: number, length: number): number[] {
  return new Array<number>(length).fill(value);
}

function playback(contextRate: number) {
  const processor = new PlaybackProcessor({ processorOptions: { sourceRate: 24000, contextRate } });
  const { port, posted } = listen(processor);
  const outputs: Float32Array[] = [];
  return {
    posted,
    queue(samples: ArrayLike<number>) {
      send(port, { type: "audio", pcm: pcm16(samples) });
    },
    control(type: "pause" | "resume" | "flush") {
      send(port, { type });
    },
    /** Renders blocks into two channels and returns channel 0. */
    render(blocks: number): Float32Array {
      const start = outputs.length;
      for (let b = 0; b < blocks; b++) {
        const left = new Float32Array(BLOCK);
        const right = new Float32Array(BLOCK);
        expect(processor.process([], [[left, right]])).toBe(true);
        expect(Array.from(right)).toEqual(Array.from(left));
        outputs.push(left);
      }
      return concat(outputs.slice(start));
    },
    playheads(): PlayheadMessage[] {
      return posted.filter((m): m is PlayheadMessage => m.type === "playhead");
    },
    drains(): DrainedMessage[] {
      return posted.filter((m): m is DrainedMessage => m.type === "drained");
    },
    lastPlayhead(): PlayheadMessage {
      const all = this.playheads();
      expect(all.length).toBeGreaterThan(0);
      return all[all.length - 1];
    },
  };
}

function nonSilent(samples: Float32Array): number {
  return samples.reduce((count, s) => (s !== 0 ? count + 1 : count), 0);
}

describe("Downsampler", () => {
  it("passes a 1 kHz sine from 48 kHz to 16 kHz at the same frequency and level", () => {
    const out = new Downsampler(48000, 16000).push(sine(1000, 48000, 48000));
    // Skip the filter warm-up, then keep a whole number of 16-sample periods.
    const settled = out.subarray(160, 160 + 16 * 900);
    let inPhase = 0;
    let quadrature = 0;
    for (let i = 0; i < settled.length; i++) {
      const angle = (2 * Math.PI * 1000 * (i + 160)) / 16000;
      inPhase += settled[i] * Math.sin(angle);
      quadrature += settled[i] * Math.cos(angle);
    }
    const amplitude = (2 / settled.length) * Math.hypot(inPhase, quadrature);
    expect(Math.abs(db(amplitude))).toBeLessThan(1);
    expect(Math.abs(db(rms(settled) / Math.SQRT1_2))).toBeLessThan(1);
    // Everything that is not the 1 kHz component is small.
    const fitted = settled.map((_, i) => {
      const angle = (2 * Math.PI * 1000 * (i + 160)) / 16000;
      return (2 / settled.length) * (inPhase * Math.sin(angle) + quadrature * Math.cos(angle));
    });
    const residual = settled.map((s, i) => s - fitted[i]);
    expect(rms(residual)).toBeLessThan(0.01);
  });

  it("removes a 10 kHz sine by at least 30 dB", () => {
    const out = new Downsampler(48000, 16000).push(sine(10000, 48000, 48000));
    expect(db(rms(out.subarray(160)) / Math.SQRT1_2)).toBeLessThan(-30);
  });

  it("filters as well at 96 and 192 kHz as at 48 kHz", () => {
    for (const rate of [96000, 192000]) {
      const out = new Downsampler(rate, 16000).push(sine(10000, rate, rate));
      expect(db(rms(out.subarray(160)) / Math.SQRT1_2)).toBeLessThan(-30);
      const kept = new Downsampler(rate, 16000).push(sine(1000, rate, rate));
      expect(Math.abs(db(rms(kept.subarray(160)) / Math.SQRT1_2))).toBeLessThan(1);
    }
  });

  it("gives the same samples for 128-sample blocks as for one call", () => {
    for (const rate of [48000, 44100, 96000]) {
      const input = noise(BLOCK * 50 + 37);
      const whole = new Downsampler(rate, 16000).push(input);
      const blocks = inBlocks(new Downsampler(rate, 16000), input);
      expect(blocks.length).toBe(whole.length);
      expect(Array.from(blocks)).toEqual(Array.from(whole));
    }
  });

  it("turns 44.1 kHz into about 16000 samples per second", () => {
    const out = inBlocks(new Downsampler(44100, 16000), sine(1000, 44100, 44100));
    const expected = (44100 * 16000) / 44100;
    expect(out.length).toBeLessThanOrEqual(expected);
    expect(out.length).toBeGreaterThan(expected - 20);
    expect(Math.abs(db(rms(out.subarray(160)) / Math.SQRT1_2))).toBeLessThan(1);
  });
});

describe("CaptureProcessor", () => {
  it("posts 80 ms chunks of 2560 bytes with an RMS level", () => {
    const processor = new CaptureProcessor({ processorOptions: { contextRate: 48000 } });
    const { posted, transfers } = listen(processor);
    const input = sine(440, 48000, 48000);
    for (let i = 0; i + BLOCK <= input.length; i += BLOCK) {
      expect(processor.process([[input.subarray(i, i + BLOCK)]])).toBe(true);
    }
    const chunks = posted.filter((m): m is ChunkMessage => m.type === "chunk");
    // One second at 16 kHz holds twelve full chunks.
    expect(chunks.length).toBe(12);
    chunks.forEach((chunk, i) => {
      expect(chunk.pcm.byteLength).toBe(2560);
      expect(transfers[i]).toEqual([chunk.pcm]);
    });
    expect(chunks[chunks.length - 1].level).toBeCloseTo(Math.SQRT1_2, 2);
    const last = new Int16Array(chunks[chunks.length - 1].pcm);
    expect(Math.max(...last)).toBeGreaterThan(32000);
  });

  it("stops processing after a stop message", () => {
    const processor = new CaptureProcessor({ processorOptions: { contextRate: 48000 } });
    const { port } = listen(processor);
    send(port, { type: "stop" });
    expect(processor.process([[new Float32Array(BLOCK)]])).toBe(false);
  });
});

describe("PlaybackProcessor", () => {
  it("plays 100 ms, reports the playhead and drains once", () => {
    const player = playback(48000);
    player.queue(constant(0.5, 2400));
    const out = player.render(40);
    expect(player.lastPlayhead()).toMatchObject({ playedMs: 100, bufferedMs: 0 });
    expect(player.drains()).toEqual([{ type: "drained", playedMs: 100 }]);
    expect(nonSilent(out)).toBe(4800);
    // Reports come at most every 50 ms of output while playing.
    expect(player.playheads().length).toBeLessThanOrEqual(3);
  });

  it("stops the playhead while paused and continues from the same place", () => {
    const player = playback(48000);
    const ramp = Array.from({ length: 4800 }, (_, i) => (i / 4800) * 0.9);
    player.queue(ramp);
    player.render(10);
    player.control("pause");
    const atPause = player.lastPlayhead().playedMs;
    // 1280 output frames consumed 640 source samples.
    expect(atPause).toBe((640 * 1000) / 24000);
    const countBefore = player.playheads().length;

    const paused = player.render(20);
    // Only the short fade-out is audible.
    expect(nonSilent(paused.subarray(240))).toBe(0);
    expect(player.playheads().length).toBe(countBefore);

    player.control("resume");
    expect(player.lastPlayhead().playedMs).toBe(atPause);
    const resumed = player.render(1);
    // After the 5 ms fade-in the output is the ramp from the paused position.
    const more = concat([resumed, player.render(3)]);
    expect(more[300]).toBeCloseTo(ramp[640 + 150], 3);

    player.render(80);
    expect(player.lastPlayhead()).toMatchObject({ playedMs: 200, bufferedMs: 0 });
    const played = player.playheads().map((m) => m.playedMs);
    expect([...played].sort((a, b) => a - b)).toEqual(played);
    expect(player.drains()).toEqual([{ type: "drained", playedMs: 200 }]);
  });

  it("drops buffered audio on flush and keeps the playhead", () => {
    const player = playback(48000);
    player.queue(constant(0.5, 4800));
    player.render(10);
    player.control("flush");
    const atFlush = (640 * 1000) / 24000;
    expect(player.lastPlayhead()).toMatchObject({ playedMs: atFlush, bufferedMs: 0 });

    const after = player.render(20);
    expect(nonSilent(after.subarray(240))).toBe(0);
    expect(player.lastPlayhead().playedMs).toBe(atFlush);
    // A flush is not a drain: the audio was dropped, not played.
    expect(player.drains()).toEqual([]);

    player.queue(constant(0.5, 2400));
    player.render(40);
    const total = ((640 + 2400) * 1000) / 24000;
    expect(player.lastPlayhead()).toMatchObject({ playedMs: total, bufferedMs: 0 });
    expect(player.drains()).toEqual([{ type: "drained", playedMs: total }]);
  });

  it("plays and drains again when audio arrives after a drain", () => {
    const player = playback(48000);
    player.queue(constant(0.5, 2400));
    player.render(40);
    player.queue(constant(-0.5, 1200));
    const second = player.render(30);
    expect(nonSilent(second)).toBe(2400);
    expect(player.drains()).toEqual([
      { type: "drained", playedMs: 100 },
      { type: "drained", playedMs: 150 },
    ]);
    expect(player.lastPlayhead()).toMatchObject({ playedMs: 150, bufferedMs: 0 });
  });

  it("keeps the played time exact across blocks at 44.1 kHz", () => {
    const player = playback(44100);
    player.queue(constant(0.5, 1200));
    const first = player.render(5);
    player.queue(constant(0.5, 1200));
    const rest = player.render(40);
    // 2400 source samples at 24000 / 44100 per output frame.
    expect(nonSilent(first) + nonSilent(rest)).toBe(4410);
    const played = player.playheads().map((m) => m.playedMs);
    for (const ms of played) {
      // Every report is a whole number of source samples.
      expect(Math.abs(ms * 24 - Math.round(ms * 24))).toBeLessThan(1e-9);
    }
    expect([...played].sort((a, b) => a - b)).toEqual(played);
    expect(player.lastPlayhead()).toMatchObject({ playedMs: 100, bufferedMs: 0 });
    expect(player.drains()).toEqual([{ type: "drained", playedMs: 100 }]);
  });
});

class FakeContext implements SuspendableContext {
  state = "running";
  resumes = 0;
  resumeFails = false;
  private listener: (() => void) | null = null;

  resume(): Promise<void> {
    this.resumes += 1;
    return this.resumeFails ? Promise.reject(new Error("not allowed")) : Promise.resolve();
  }
  addEventListener(_type: "statechange", listener: () => void): void {
    this.listener = listener;
  }
  removeEventListener(): void {
    this.listener = null;
  }
  /** The browser moves the context to another state. */
  become(state: string): void {
    this.state = state;
    this.listener?.();
  }
}

class FakePage implements PageVisibility {
  shown = true;
  private onChange: (() => void) | null = null;

  visible(): boolean {
    return this.shown;
  }
  watch(onChange: () => void): () => void {
    this.onChange = onChange;
    return () => {
      this.onChange = null;
    };
  }
  show(shown: boolean): void {
    this.shown = shown;
    this.onChange?.();
  }
}

describe("keepRunning", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("resumes a suspended context and gives up when it stays suspended for 5 s", () => {
    const context = new FakeContext();
    const gaveUp = vi.fn();
    keepRunning(context, new FakePage(), gaveUp);
    context.become("suspended");
    expect(context.resumes).toBe(1);
    vi.advanceTimersByTime(4999);
    expect(gaveUp).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1);
    expect(gaveUp).toHaveBeenCalledTimes(1);
    context.become("suspended"); // it no longer watches
    expect(context.resumes).toBe(1);
  });

  it("waits while the page is hidden, and is content once the context runs again", () => {
    const context = new FakeContext();
    const page = new FakePage();
    const gaveUp = vi.fn();
    page.shown = false;
    keepRunning(context, page, gaveUp);
    context.become("interrupted");
    vi.advanceTimersByTime(60_000);
    expect([context.resumes, gaveUp.mock.calls.length]).toEqual([0, 0]);
    page.show(true);
    expect(context.resumes).toBe(1);
    context.become("running");
    vi.advanceTimersByTime(60_000);
    expect(gaveUp).not.toHaveBeenCalled();
  });

  it("gives up when resuming fails", async () => {
    const context = new FakeContext();
    context.resumeFails = true;
    const gaveUp = vi.fn();
    keepRunning(context, new FakePage(), gaveUp);
    context.become("suspended");
    await Promise.resolve();
    expect(gaveUp).toHaveBeenCalledTimes(1);
  });

  it("does nothing once it is stopped", () => {
    const context = new FakeContext();
    const gaveUp = vi.fn();
    const stop = keepRunning(context, new FakePage(), gaveUp);
    stop();
    context.become("suspended");
    vi.advanceTimersByTime(60_000);
    expect([context.resumes, gaveUp.mock.calls.length]).toEqual([0, 0]);
  });
});
