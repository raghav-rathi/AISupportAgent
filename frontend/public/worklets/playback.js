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
// The server follows every pause with a resume or a flush within about 0.9 s. A pause that
// outlives this was forgotten, and would otherwise mute every later reply.
const PAUSE_LIMIT_SECONDS = 1.5;
// The most audio kept waiting. More only arrives while the player is held up, and then it would
// grow without bound.
const MAX_BUFFER_SECONDS = 60;
// Cutoff of the filter that runs before the context's slower rate, as a fraction of that rate.
const LOWPASS_FRACTION = 0.45;

// One second-order Butterworth low-pass section, by the bilinear transform. Two in a row make
// a fourth-order filter.
function section(cutoff, rate, q) {
  const w = (2 * Math.PI * cutoff) / rate;
  const cos = Math.cos(w);
  const alpha = Math.sin(w) / (2 * q);
  const a0 = 1 + alpha;
  return {
    b0: (1 - cos) / 2 / a0,
    b1: (1 - cos) / a0,
    b2: (1 - cos) / 2 / a0,
    a1: (-2 * cos) / a0,
    a2: (1 - alpha) / a0,
    z1: 0,
    z2: 0,
  };
}

function step(s, x) {
  const y = s.b0 * x + s.z1;
  s.z1 = s.b1 * x - s.a1 * y + s.z2;
  s.z2 = s.b2 * x - s.a2 * y;
  return y;
}

class PlaybackProcessor extends Base {
  constructor(options) {
    super();
    const opts = options?.processorOptions ?? {};
    this.contextRate = typeof globalThis.sampleRate === "number" ? globalThis.sampleRate : (opts.contextRate ?? 48000);
    this.rampLength = Math.max(1, Math.round(this.contextRate * RAMP_SECONDS));
    this.reportFrames = Math.round(this.contextRate * REPORT_SECONDS);
    this.pauseLimit = Math.round(this.contextRate * PAUSE_LIMIT_SECONDS);
    this.setSourceRate(opts.sourceRate ?? 24000);
    this.head = 0;
    this.tail = 0;
    // Read position past `head` in units of 1 / contextRate. Integer steps keep
    // the playhead exact for ratios like 24000 / 44100.
    this.phase = 0;
    this.consumed = 0;
    this.epoch = 0; // the number of the server's last flush
    this.paused = false;
    this.pausedFrames = 0;
    this.inRun = false;
    this.gain = 0;
    this.fade = null;
    this.fadeIndex = 0;
    this.sinceReport = 0;
    this.level = 0;
    this.stopped = false;
    this.port.onmessage = (event) => this.handle(event.data);
  }

  // Sets the rate of the audio that arrives. Only called before any has.
  setSourceRate(rate) {
    this.sourceRate = rate;
    this.queue = new Float32Array(rate);
    this.maxQueue = rate * MAX_BUFFER_SECONDS;
    // Linear interpolation lets everything above the context's Nyquist fold back into the voice,
    // so a slower context, like a 16 kHz Bluetooth headset, gets it filtered out first.
    if (this.contextRate < rate) {
      const cutoff = LOWPASS_FRACTION * this.contextRate;
      this.lowpass = [section(cutoff, rate, 0.5412), section(cutoff, rate, 1.3066)];
    } else {
      this.lowpass = null;
    }
  }

  handle(message) {
    switch (message?.type) {
      case "audio":
        this.enqueue(message.pcm);
        break;
      case "rates":
        if (message.sourceRate > 0 && message.sourceRate !== this.sourceRate) this.setSourceRate(message.sourceRate);
        break;
      case "pause":
        if (!this.paused) {
          this.startFade();
          this.paused = true;
        }
        this.pausedFrames = 0;
        this.report("pause");
        break;
      case "resume":
        if (this.paused) {
          this.paused = false;
          this.gain = 0;
        }
        this.report("resume");
        break;
      case "flush": {
        // Drops what is queued and ends a pause: what comes next is to be heard. A flush with a
        // number is the server's. One without is the visitor's Stop, answered as "stop".
        const own = typeof message.epoch !== "number";
        if (!own) this.epoch = message.epoch;
        this.startFade();
        this.paused = false;
        this.head = 0;
        this.tail = 0;
        this.phase = 0;
        this.inRun = false;
        this.sinceReport = 0;
        // A long reply may have grown the queue. What it held is gone, so its memory can go too.
        if (this.queue.length > this.sourceRate) this.queue = new Float32Array(this.sourceRate);
        if (this.lowpass) for (const s of this.lowpass) s.z1 = s.z2 = 0;
        this.report(own ? "stop" : "flush");
        break;
      }
      case "stop":
        this.stopped = true;
        this.queue = new Float32Array(0);
        this.head = 0;
        this.tail = 0;
        break;
    }
  }

  enqueue(pcm) {
    if (this.stopped) return;
    // Browsers run on little endian hardware, so Int16Array reads PCM16 LE directly.
    const samples = new Int16Array(pcm, 0, pcm.byteLength >> 1);
    const count = this.reserve(samples.length);
    if (count === 0) return;
    const lowpass = this.lowpass;
    for (let i = 0; i < count; i++) {
      let value = samples[i] / 32768;
      if (lowpass) value = step(lowpass[1], step(lowpass[0], value));
      this.queue[this.tail + i] = value;
    }
    this.tail += count;
    this.inRun = true;
  }

  // Makes room for `count` more samples, as far as the cap allows, and says how many fit.
  reserve(count) {
    const live = this.tail - this.head;
    const fits = Math.max(0, Math.min(count, this.maxQueue - live));
    if (this.tail + fits <= this.queue.length) return fits;
    if (live + fits <= this.queue.length) {
      this.queue.copyWithin(0, this.head, this.tail);
    } else {
      const next = new Float32Array(Math.min(this.maxQueue, Math.max(this.queue.length * 2, live + fits)));
      next.set(this.queue.subarray(this.head, this.tail));
      this.queue = next;
    }
    this.head = 0;
    this.tail = live;
    return fits;
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
  // `epoch` says which of the server's flushes the playhead comes after.
  report(after = "tick") {
    this.port.postMessage({
      type: "playhead",
      playedMs: this.playedMs(),
      bufferedMs: ((this.tail - this.head) * 1000) / this.sourceRate,
      level: this.level,
      after,
      epoch: this.epoch,
    });
  }

  process(inputs, outputs) {
    if (this.stopped) return false;
    const output = outputs[0];
    if (!output || output.length === 0) return true;
    const channel = output[0];
    const frames = channel.length;
    let sumSquares = 0;
    let played = 0;
    let drained = false;
    if (this.paused) {
      this.pausedFrames += frames;
      if (this.pausedFrames >= this.pauseLimit) {
        this.paused = false;
        this.gain = 0;
        this.report("resume");
      }
    }
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
          // The gain stays: when the network was only late, the audio that follows goes on at
          // full level instead of fading in over its first milliseconds.
          this.head = 0;
          this.tail = 0;
          this.phase = 0;
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
