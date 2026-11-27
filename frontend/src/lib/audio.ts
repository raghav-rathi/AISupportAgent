// Audio engine for voice mode: microphone capture as 16 kHz PCM16 chunks and
// playback of 24 kHz PCM16 with an exact playhead. Both run in AudioWorklets
// loaded from /public/worklets.

export const CAPTURE_RATE = 16000;
export const PLAYBACK_RATE = 24000;
/** 80 ms of 16 kHz mono PCM16. */
export const CHUNK_BYTES = 2560;

/** What a playhead report follows: the regular tick, a command, or the end of the audio. */
export type PlayheadCause = "tick" | "pause" | "resume" | "flush" | "drained";

export interface VoiceAudioEvents {
  /** 16 kHz mono PCM16, exactly 2560 bytes (80 ms) per call. */
  onChunk(pcm: ArrayBuffer): void;
  /** Microphone level 0..1, once per chunk. */
  onMicLevel(level: number): void;
  /** Level 0..1 of the audio being played, with each playhead report. */
  onOutLevel(level: number): void;
  /** Milliseconds of received audio played since start(), and milliseconds still buffered. */
  onPlayhead(playedMs: number, bufferedMs: number, after: PlayheadCause): void;
  /** Everything queued has been played. Fires once per run of audio. */
  onDrained(playedMs: number): void;
}

export interface VoiceAudioInfo {
  echoCancellation: boolean | string | null;
  contextRate: number;
}

type CaptureMessage = { type: "chunk"; pcm: ArrayBuffer; level: number };

type PlaybackMessage =
  | { type: "playhead"; playedMs: number; bufferedMs: number; level: number; after: PlayheadCause }
  | { type: "drained"; playedMs: number };

const MIC_SETTINGS = { channelCount: 1, noiseSuppression: false, autoGainControl: true };

async function openMicrophone(): Promise<MediaStream> {
  // "all" asks for echo cancellation against all system audio where the
  // browser supports it. The DOM types only know booleans.
  const preferred = { ...MIC_SETTINGS, echoCancellation: "all" } as unknown as MediaTrackConstraints;
  try {
    return await navigator.mediaDevices.getUserMedia({ audio: preferred });
  } catch (error) {
    // A denied permission would only be denied again.
    if (error instanceof DOMException && (error.name === "NotAllowedError" || error.name === "SecurityError")) {
      throw error;
    }
    return navigator.mediaDevices.getUserMedia({ audio: { ...MIC_SETTINGS, echoCancellation: true } });
  }
}

export class VoiceAudio {
  private context: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private capture: AudioWorkletNode | null = null;
  private mute: GainNode | null = null;
  private playback: AudioWorkletNode | null = null;
  private played = 0;
  private stopped = true;

  static supported(): boolean {
    return (
      typeof AudioWorkletNode === "function" &&
      typeof navigator !== "undefined" &&
      typeof navigator.mediaDevices?.getUserMedia === "function"
    );
  }

  async start(events: VoiceAudioEvents): Promise<VoiceAudioInfo> {
    if (!this.stopped) throw new Error("VoiceAudio is already started");
    this.stopped = false;
    this.played = 0;
    try {
      // The microphone opens first so no playback node exists before it.
      this.stream = await openMicrophone();
      this.checkRunning();

      const context = new AudioContext({ latencyHint: "interactive" });
      this.context = context;
      await Promise.all([
        context.audioWorklet.addModule("/worklets/capture.js"),
        context.audioWorklet.addModule("/worklets/playback.js"),
      ]);
      this.checkRunning();

      this.source = context.createMediaStreamSource(this.stream);
      this.capture = new AudioWorkletNode(context, "capture", {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [1],
        channelCount: 1,
        channelCountMode: "explicit",
        processorOptions: { contextRate: context.sampleRate },
      });
      this.capture.port.onmessage = (event: MessageEvent<CaptureMessage>) => {
        if (this.stopped || event.data.type !== "chunk") return;
        events.onMicLevel(event.data.level);
        events.onChunk(event.data.pcm);
      };
      // The capture node only runs while something pulls its output, so it
      // feeds the destination through a silent gain.
      this.mute = context.createGain();
      this.mute.gain.value = 0;
      this.source.connect(this.capture).connect(this.mute).connect(context.destination);

      this.playback = new AudioWorkletNode(context, "playback", {
        numberOfInputs: 0,
        numberOfOutputs: 1,
        outputChannelCount: [1],
        processorOptions: { sourceRate: PLAYBACK_RATE, contextRate: context.sampleRate },
      });
      this.playback.port.onmessage = (event: MessageEvent<PlaybackMessage>) => {
        if (this.stopped) return;
        const message = event.data;
        if (message.type === "playhead") {
          this.played = Math.max(this.played, message.playedMs);
          events.onOutLevel(message.level);
          events.onPlayhead(message.playedMs, message.bufferedMs, message.after);
        } else if (message.type === "drained") {
          this.played = Math.max(this.played, message.playedMs);
          events.onDrained(message.playedMs);
        }
      };
      this.playback.connect(context.destination);

      await context.resume();
      this.checkRunning();

      const track = this.stream.getAudioTracks()[0];
      const reported: unknown = track?.getSettings().echoCancellation;
      return {
        echoCancellation: typeof reported === "boolean" || typeof reported === "string" ? reported : null,
        contextRate: context.sampleRate,
      };
    } catch (error) {
      this.stopped = true;
      await this.release();
      throw error;
    }
  }

  /** Queues 24 kHz mono PCM16. The buffer is transferred to the worklet. */
  play(pcm: ArrayBuffer): void {
    if (this.stopped || !this.playback || pcm.byteLength === 0) return;
    this.playback.port.postMessage({ type: "audio", pcm }, [pcm]);
  }

  pause(): void {
    this.send({ type: "pause" });
  }

  resume(): void {
    this.send({ type: "resume" });
  }

  /** Drops buffered audio and ends a pause. The playhead keeps its value. */
  flush(): void {
    this.send({ type: "flush" });
  }

  playedMs(): number {
    return this.played;
  }

  async stop(): Promise<void> {
    this.stopped = true;
    await this.release();
  }

  private send(message: { type: "pause" | "resume" | "flush" }): void {
    if (this.stopped || !this.playback) return;
    this.playback.port.postMessage(message);
  }

  // stop() may run while start() awaits; start() then gives up here.
  private checkRunning(): void {
    if (this.stopped) throw new Error("VoiceAudio was stopped while starting");
  }

  private async release(): Promise<void> {
    const { context, stream, source, capture, mute, playback } = this;
    this.context = null;
    this.stream = null;
    this.source = null;
    this.capture = null;
    this.mute = null;
    this.playback = null;

    if (capture) {
      capture.port.postMessage({ type: "stop" });
      capture.port.onmessage = null;
    }
    if (playback) playback.port.onmessage = null;
    stream?.getTracks().forEach((track) => track.stop());
    source?.disconnect();
    capture?.disconnect();
    mute?.disconnect();
    playback?.disconnect();
    if (context && context.state !== "closed") {
      try {
        await context.close();
      } catch {
        // Closing can fail if the context already died; nothing is left to release.
      }
    }
  }
}
