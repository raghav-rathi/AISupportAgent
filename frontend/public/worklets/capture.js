// Microphone capture worklet: downsamples the context rate to 16 kHz mono PCM16
// and posts one 80 ms chunk at a time.

// The fallback base class lets the file load in Node for tests.
const Base =
  globalThis.AudioWorkletProcessor ??
  class {
    constructor() {
      this.port = { postMessage() {}, onmessage: null };
    }
  };

const OUTPUT_RATE = 16000;
const CHUNK_SAMPLES = 1280;
// Filter length in input samples at 48 kHz. Higher input rates get proportionally more taps, so
// the filter spans the same time and keeps its sharpness at 96 or 192 kHz.
const TAPS = 32;
const TAPS_RATIO = 3; // 48 kHz to 16 kHz

function sinc(x) {
  if (x === 0) return 1;
  const px = Math.PI * x;
  return Math.sin(px) / px;
}

// Resamples by any ratio with a Hann windowed-sinc low-pass evaluated at the
// exact fractional input position of each output sample. Output positions are
// computed from the output count, so the result does not depend on how the
// input is split into blocks.
class Downsampler {
  constructor(inputRate, outputRate = OUTPUT_RATE) {
    this.ratio = inputRate / outputRate;
    // 44.1 and 48 kHz keep 32 taps. Above that the count grows with the ratio.
    this.half = Math.max(TAPS / 2, Math.ceil(((TAPS / 2) * this.ratio) / TAPS_RATIO));
    // Cutoff just under the lower Nyquist, as a fraction of the input rate times two.
    this.band = (2 * 0.45 * Math.min(inputRate, outputRate)) / inputRate;
    // The buffer starts with zeros so the first outputs see silence before the input.
    this.buffer = new Float32Array(1024);
    this.length = this.half;
    this.start = -this.half;
    this.produced = 0;
  }

  push(input) {
    this.append(input);
    const end = this.start + this.length;
    let count = 0;
    while (Math.floor((this.produced + count) * this.ratio) + this.half < end) count++;
    const out = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      out[i] = this.sampleAt((this.produced + i) * this.ratio);
    }
    this.produced += count;
    this.trim();
    return out;
  }

  append(input) {
    const needed = this.length + input.length;
    if (needed > this.buffer.length) {
      const next = new Float32Array(Math.max(needed, this.buffer.length * 2));
      next.set(this.buffer.subarray(0, this.length));
      this.buffer = next;
    }
    this.buffer.set(input, this.length);
    this.length = needed;
  }

  sampleAt(t) {
    const base = Math.floor(t);
    let sum = 0;
    let weights = 0;
    for (let k = base - this.half + 1; k <= base + this.half; k++) {
      const d = t - k;
      const w = this.band * sinc(this.band * d) * (0.5 + 0.5 * Math.cos((Math.PI * d) / this.half));
      weights += w;
      const index = k - this.start;
      if (index >= 0) sum += w * this.buffer[index];
    }
    // Normalizing per output keeps unity gain at DC for every fractional phase.
    return sum / weights;
  }

  // Drops input that no future output can reach.
  trim() {
    const keepFrom = Math.floor(this.produced * this.ratio) - this.half + 1;
    const drop = Math.min(this.length, keepFrom - this.start);
    if (drop <= 0) return;
    this.buffer.copyWithin(0, drop, this.length);
    this.length -= drop;
    this.start += drop;
  }
}

class CaptureProcessor extends Base {
  constructor(options) {
    super();
    const opts = options?.processorOptions ?? {};
    const rate = typeof globalThis.sampleRate === "number" ? globalThis.sampleRate : (opts.contextRate ?? 48000);
    this.downsampler = new Downsampler(rate, OUTPUT_RATE);
    this.chunk = new Int16Array(CHUNK_SAMPLES);
    this.filled = 0;
    this.sumSquares = 0;
    this.stopped = false;
    this.port.onmessage = (event) => {
      if (event.data?.type === "stop") this.stopped = true;
    };
  }

  process(inputs) {
    if (this.stopped) return false;
    const channel = inputs[0]?.[0];
    // No channel means the microphone is not connected yet.
    if (!channel) return true;
    const samples = this.downsampler.push(channel);
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      this.chunk[this.filled++] = Math.round(s * 32767);
      this.sumSquares += s * s;
      if (this.filled === CHUNK_SAMPLES) this.post();
    }
    return true;
  }

  post() {
    const pcm = this.chunk.buffer;
    const level = Math.min(1, Math.sqrt(this.sumSquares / CHUNK_SAMPLES));
    this.port.postMessage({ type: "chunk", pcm, level }, [pcm]);
    // The old buffer was transferred, so each chunk needs a new one.
    this.chunk = new Int16Array(CHUNK_SAMPLES);
    this.filled = 0;
    this.sumSquares = 0;
  }
}

if (typeof globalThis.registerProcessor === "function") {
  globalThis.registerProcessor("capture", CaptureProcessor);
}

export { Downsampler, CaptureProcessor };
