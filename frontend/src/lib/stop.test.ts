import { describe, expect, it } from "vitest";

import type { ServerEvent } from "./protocol";
import { type AppState, initialState, reduce } from "./state";
import { type StopKeyPress, stopKey, stopTarget } from "./stop";

const press = (key: string, more: Partial<StopKeyPress> = {}): StopKeyPress => ({
  key,
  repeat: false,
  modified: false,
  defaultPrevented: false,
  focus: "none",
  fieldHasSelection: false,
  dialogOpen: false,
  ...more,
});

describe("the Stop keys", () => {
  it("take the space bar from a button the pointer left focused, so it does not end voice", () => {
    expect(stopKey(press(" ", { focus: "pointer-button" }), true)).toBe("stop");
    // With nothing to stop the button still must not toggle, and the page must not scroll.
    expect(stopKey(press(" ", { focus: "pointer-button" }), false)).toBe("block");
    expect(stopKey(press(" "), true)).toBe("stop");
  });

  it("leave the space bar to a control the keyboard focused, and to text fields", () => {
    expect(stopKey(press(" ", { focus: "control" }), true)).toBe("pass");
    expect(stopKey(press(" ", { focus: "field" }), true)).toBe("pass");
  });

  it("stop on Escape from anywhere, a text field too", () => {
    expect(stopKey(press("Escape"), true)).toBe("stop");
    expect(stopKey(press("Escape", { focus: "field" }), true)).toBe("stop");
    expect(stopKey(press("Escape", { focus: "control" }), true)).toBe("stop");
  });

  it("leave Escape to a text field with a selection, to an open dialog or menu, and when there is nothing to stop", () => {
    expect(stopKey(press("Escape", { focus: "field", fieldHasSelection: true }), true)).toBe("pass");
    expect(stopKey(press("Escape", { dialogOpen: true }), true)).toBe("pass");
    expect(stopKey(press("Escape"), false)).toBe("pass");
  });

  it("ignore held keys, shortcuts, keys handled already, and other keys", () => {
    expect(stopKey(press(" ", { repeat: true }), true)).toBe("pass");
    expect(stopKey(press(" ", { modified: true }), true)).toBe("pass");
    expect(stopKey(press("Escape", { defaultPrevented: true }), true)).toBe("pass");
    expect(stopKey(press("Enter"), true)).toBe("pass");
  });
});

const at = (event: Record<string, unknown>, seq: number) => ({ seq, ts: 100 + seq, ...event }) as ServerEvent;

function stateWith(...events: ServerEvent[]): AppState {
  let state = reduce(initialState, {
    type: "event",
    event: at(
      {
        type: "session.ready",
        turn_id: null,
        session_id: "s1",
        last_seq: 0,
        customer: null,
        tickets: [],
        busy: false,
        resumable: false,
        fault_injection: false,
        voice: true,
      },
      0,
    ),
  });
  state = reduce(state, { type: "event", event: at({ type: "session.caught_up", turn_id: null }, 0) });
  for (const event of events) state = reduce(state, { type: "event", event });
  return state;
}

const voiceOn = (state: AppState) => reduce(state, { type: "voice", status: "on" });
const activity = (state: AppState, value: string) =>
  reduce(state, { type: "event", event: at({ type: "voice.state", turn_id: null, state: value }, 0) });
const turn = (mode: "text" | "voice", seq: number) => at({ type: "turn.started", turn_id: "t1", mode, text: "Hi", client_id: null }, seq);

describe("what Stop stops", () => {
  it("cancels a typed turn as a turn, also while voice is on", () => {
    expect(stopTarget(voiceOn(stateWith(turn("text", 1))))).toBe("turn");
  });

  it("interrupts a spoken turn, a reply being spoken, and a voice turn being thought about", () => {
    expect(stopTarget(voiceOn(stateWith(turn("voice", 1))))).toBe("reply");
    expect(stopTarget(activity(voiceOn(stateWith()), "speaking"))).toBe("reply");
    expect(stopTarget(activity(voiceOn(stateWith()), "thinking"))).toBe("reply");
  });

  it("has nothing to stop while voice listens and no turn runs", () => {
    expect(stopTarget(voiceOn(stateWith()))).toBeNull();
    expect(stopTarget(activity(voiceOn(stateWith()), "hearing"))).toBeNull();
  });

  it("cancels a spoken turn as a turn while voice is not on to take an interrupt", () => {
    expect(stopTarget(stateWith(turn("voice", 1)))).toBe("turn");
  });
});
