// Playback worklet: plays queued mono PCM16 at the source rate, resampled to
// the context rate, and reports an exact playhead in source samples.

// The fallback base class lets the file load in Node for tests.
const Base =
  globalThis.AudioWorkletProcessor ??
  class {
    constructor() {
      this.port = { postMessage() {}, onmessage: null };
    }
  };

const RAMP_SECONDS = 0.005;
const REPORT_SECONDS = 0.05;

class PlaybackProcessor extends Base {
  constructor(options) {
    super();
    const opts = options?.processorOptions ?? {};
    this.sourceRate = opts.sourceRate ?? 24000;
    this.contextRate = typeof globalThis.sampleRate === "number" ? globalThis.sampleRate : (opts.contextRate ?? 48000);
    this.rampLength = Math.max(1, Math.round(this.contextRate * RAMP_SECONDS));
    this.reportFrames = Math.round(this.contextRate * REPORT_SECONDS);
    this.queue = new Float32Array(this.sourceRate);
    this.head = 0;
    this.tail = 0;
    // Read position past `head` in units of 1 / contextRate. Integer steps keep
    // the playhead exact for ratios like 24000 / 44100.
    this.phase = 0;
    this.consumed = 0;
    this.paused = false;
    this.inRun = false;
    this.gain = 0;
    this.fade = null;
    this.fadeIndex = 0;
    this.sinceReport = 0;
    this.level = 0;
    this.port.onmessage = (event) => this.handle(event.data);
  }

  handle(message) {
    switch (message?.type) {
      case "audio":
        this.enqueue(message.pcm);
        break;
      case "pause":
        if (!this.paused) {
          this.startFade();
          this.paused = true;
        }
        this.report("pause");
        break;
      case "resume":
        if (this.paused) {
          this.paused = false;
          this.gain = 0;
        }
        this.report("resume");
        break;
      case "flush":
        // Drops what is queued and ends a pause: what comes next is to be heard.
        this.startFade();
        this.paused = false;
        this.head = 0;
        this.tail = 0;
        this.phase = 0;
        this.inRun = false;
        this.sinceReport = 0;
        this.report("flush");
        break;
    }
  }

  enqueue(pcm) {
    // Browsers run on little endian hardware, so Int16Array reads PCM16 LE directly.
    const samples = new Int16Array(pcm, 0, pcm.byteLength >> 1);
    const count = samples.length;
    if (count === 0) return;
    this.reserve(count);
    for (let i = 0; i < count; i++) this.queue[this.tail + i] = samples[i] / 32768;
    this.tail += count;
    this.inRun = true;
  }

  reserve(count) {
    if (this.tail + count <= this.queue.length) return;
    const live = this.tail - this.head;
    if (live + count <= this.queue.length) {
      this.queue.copyWithin(0, this.head, this.tail);
    } else {
      const next = new Float32Array(Math.max(this.queue.length * 2, live + count));
      next.set(this.queue.subarray(this.head, this.tail));
      this.queue = next;
    }
    this.head = 0;
    this.tail = live;
  }

  sampleAt(head, phase) {
    const a = this.queue[head];
    // The last queued sample fades toward silence instead of waiting for more audio.
    const b = head + 1 < this.tail ? this.queue[head + 1] : 0;
    return a + (b - a) * (phase / this.contextRate);
  }

  // Renders a short fade-out from the current position without consuming the
  // queue, so pause and flush do not click and the playhead stays put.
  startFade() {
    const length = this.rampLength;
    const fade = new Float32Array(length);
    if (this.fade) {
      for (let i = 0; i < length && this.fadeIndex + i < this.fade.length; i++) {
        fade[i] = this.fade[this.fadeIndex + i];
      }
    }
    if (!this.paused && this.gain > 0) {
      let head = this.head;
      let phase = this.phase;
      for (let i = 0; i < length && head < this.tail; i++) {
        fade[i] += this.gain * (1 - (i + 1) / length) * this.sampleAt(head, phase);
        phase += this.sourceRate;
        while (phase >= this.contextRate) {
          phase -= this.contextRate;
          head++;
        }
      }
    }
    this.fade = fade;
    this.fadeIndex = 0;
    this.gain = 0;
  }

  playedMs() {
    return (this.consumed * 1000) / this.sourceRate;
  }

  // `after` says what the report follows: the regular tick, a command, or the end of the audio.
  report(after = "tick") {
    this.port.postMessage({
      type: "playhead",
      playedMs: this.playedMs(),
      bufferedMs: ((this.tail - this.head) * 1000) / this.sourceRate,
      level: this.level,
      after,
    });
  }

  process(inputs, outputs) {
    const output = outputs[0];
    if (!output || output.length === 0) return true;
    const channel = output[0];
    const frames = channel.length;
    let sumSquares = 0;
    let played = 0;
    let drained = false;
    for (let i = 0; i < frames; i++) {
      let value = 0;
      if (this.fade) {
        value += this.fade[this.fadeIndex++];
        if (this.fadeIndex >= this.fade.length) this.fade = null;
      }
      if (!this.paused && this.head < this.tail) {
        // The gain steps before use so the first sample of a run is not silent.
        if (this.gain < 1) this.gain = Math.min(1, this.gain + 1 / this.rampLength);
        value += this.gain * this.sampleAt(this.head, this.phase);
        played++;
        this.phase += this.sourceRate;
        while (this.phase >= this.contextRate && this.head < this.tail) {
          this.phase -= this.contextRate;
          this.head++;
          this.consumed++;
        }
        if (this.head >= this.tail) {
          this.head = 0;
          this.tail = 0;
          this.phase = 0;
          this.gain = 0;
          drained = this.inRun;
          this.inRun = false;
        }
      }
      channel[i] = value;
      sumSquares += value * value;
    }
    for (let c = 1; c < output.length; c++) output[c].set(channel);
    this.level = frames > 0 ? Math.min(1, Math.sqrt(sumSquares / frames)) : 0;

    if (drained) {
      this.sinceReport = 0;
      this.report("drained");
      this.port.postMessage({ type: "drained", playedMs: this.playedMs() });
    } else if (played > 0) {
      this.sinceReport += played;
      if (this.sinceReport >= this.reportFrames) {
        this.sinceReport = 0;
        this.report();
      }
    }
    return true;
  }
}

if (typeof globalThis.registerProcessor === "function") {
  globalThis.registerProcessor("playback", PlaybackProcessor);
}

export { PlaybackProcessor };
