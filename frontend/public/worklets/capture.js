// Microphone capture worklet: downsamples the context rate to 16 kHz mono PCM16,
// or the rate the server asks for, and posts one 80 ms chunk at a time.

// The fallback base class lets the file load in Node for tests.
const Base =
  globalThis.AudioWorkletProcessor ??
  class {
    constructor() {
      this.port = { postMessage() {}, onmessage: null };
    }
  };

const OUTPUT_RATE = 16000;
const CHUNK_SECONDS = 0.08; // 80 ms
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
    // Reused for every block: the audio thread should not allocate while it runs.
    this.output = new Float32Array(256);
  }

  // Resamples one block. The new samples are the first ones of `output`, as many as returned.
  push(input) {
    this.append(input);
    const end = this.start + this.length;
    let count = 0;
    while (Math.floor((this.produced + count) * this.ratio) + this.half < end) count++;
    if (count > this.output.length) this.output = new Float32Array(Math.max(count, this.output.length * 2));
    const out = this.output;
    for (let i = 0; i < count; i++) {
      out[i] = this.sampleAt((this.produced + i) * this.ratio);
    }
    this.produced += count;
    this.trim();
    return count;
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
    this.rate = typeof globalThis.sampleRate === "number" ? globalThis.sampleRate : (opts.contextRate ?? 48000);
    this.setOutputRate(OUTPUT_RATE);
    this.filled = 0;
    this.sumSquares = 0;
    this.stopped = false;
    this.port.onmessage = (event) => {
      const message = event.data;
      if (message?.type === "stop") this.stopped = true;
      // The rate the server asked for in its answer to voice.start. It comes before any chunk is sent.
      if (message?.type === "rates" && message.outputRate > 0 && message.outputRate !== this.outputRate) {
        this.setOutputRate(message.outputRate);
      }
    };
  }

  // Starts over at a new output rate. Whatever part of a chunk was collected goes.
  setOutputRate(rate) {
    this.outputRate = rate;
    this.chunkSamples = Math.round(rate * CHUNK_SECONDS);
    this.downsampler = new Downsampler(this.rate, rate);
    this.chunk = new Int16Array(this.chunkSamples);
    this.filled = 0;
    this.sumSquares = 0;
  }

  process(inputs) {
    if (this.stopped) return false;
    const channel = inputs[0]?.[0];
    // No channel means the microphone is not connected yet.
    if (!channel) return true;
    const count = this.downsampler.push(channel);
    const samples = this.downsampler.output;
    for (let i = 0; i < count; i++) {
      const sample = samples[i];
      // A NaN from a broken input would turn the level into NaN for good. It counts as silence.
      const s = sample === sample ? Math.max(-1, Math.min(1, sample)) : 0;
      this.chunk[this.filled++] = Math.round(s * 32767);
      this.sumSquares += s * s;
      if (this.filled === this.chunkSamples) this.post();
    }
    return true;
  }

  post() {
    const pcm = this.chunk.buffer;
    const level = Math.min(1, Math.sqrt(this.sumSquares / this.chunkSamples));
    this.port.postMessage({ type: "chunk", pcm, level }, [pcm]);
    // The old buffer was transferred, so each chunk needs a new one.
    this.chunk = new Int16Array(this.chunkSamples);
    this.filled = 0;
    this.sumSquares = 0;
  }
}

if (typeof globalThis.registerProcessor === "function") {
  globalThis.registerProcessor("capture", CaptureProcessor);
}

export { Downsampler, CaptureProcessor };
