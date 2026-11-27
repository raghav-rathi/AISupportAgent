/**
 * What the one Stop of the voice panel stops, and which key presses count as pressing it. Pure, so
 * the rules can be tested without a page.
 */

import type { AppState } from "./state";

/** `turn`: a running turn, cancelled with `turn.cancel`. `reply`: the voice reply, interrupted. */
export type StopTarget = "turn" | "reply";

/** What Stop would stop now, or null when there is nothing to stop. */
export function stopTarget(state: AppState): StopTarget | null {
  const turn = state.turns.at(-1);
  const running = state.busy && turn?.status === "running";
  const voiceOn = state.voice.status === "on";
  // A typed turn is stopped as a turn. So is any turn while voice cannot take an interrupt.
  if (running && (turn?.mode === "text" || !voiceOn)) return "turn";
  const activity = voiceOn ? state.voice.activity : null;
  return running || activity === "speaking" || activity === "thinking" ? "reply" : null;
}

/** Where the keyboard focus was when a key went down. */
export type KeyFocus =
  | "none" // nothing that takes keys: the page or plain text
  | "pointer-button" // a button the pointer focused, which shows no focus ring
  | "control" // a button, link or tab the keyboard focused
  | "field"; // a text field, select or editable region

export interface StopKeyPress {
  key: string; // KeyboardEvent.key
  repeat: boolean;
  modified: boolean; // Ctrl, Alt or Meta held
  defaultPrevented: boolean;
  focus: KeyFocus;
  fieldHasSelection: boolean; // the focused text field has text selected
  dialogOpen: boolean; // an open dialog or menu takes Escape for itself
}

/**
 * `stop`: prevent the default and stop. `block`: prevent the default and do nothing else, so the
 * space bar neither scrolls nor activates a button the pointer left focused. `pass`: leave it.
 */
export type StopKeyDecision = "stop" | "block" | "pass";

/** What a key press does while voice is on. `canStop` says whether Stop is enabled. */
export function stopKey(press: StopKeyPress, canStop: boolean): StopKeyDecision {
  if (press.repeat || press.modified || press.defaultPrevented) return "pass";
  if (press.key === " ") {
    // A button the pointer focused would toggle on the space bar, which nobody aiming at Stop
    // wants. A keyboard user who focused a control gets its own action.
    if (press.focus !== "none" && press.focus !== "pointer-button") return "pass";
    return canStop ? "stop" : "block";
  }
  if (press.key === "Escape") {
    if (press.dialogOpen || press.fieldHasSelection || !canStop) return "pass";
    return "stop";
  }
  return "pass";
}
