/**
 * What to do when the socket closes. The server says why with a close code, and the answer differs:
 * a dropped network wants a quick retry, a full server wants patience, and a refusal on policy
 * wants no retry at all. Knocking every few seconds on a door that said "come back later" is how
 * one tab turns into hundreds of connections an hour.
 */

export type Plan =
  | { kind: "retry"; delayMs: number } // connect again after this long
  | { kind: "pause" } // wait until the visitor is back, then connect
  | { kind: "stop"; why: string } // do not connect again unless the visitor asks
  | { kind: "restart" }; // the conversation is gone: start a new one

/** The server's close codes, as in backend/src/supportagent/api/app.py. */
export const CLOSE = {
  policy: 1008,
  fellBehind: 4000,
  idle: 4001,
  tooManyConversations: 4002,
  atCapacity: 4003,
  tooManyConnections: 4004,
  sessionExpired: 4005,
} as const;

const QUICK_STEPS_MS = [500, 1_000, 2_000, 4_000, 8_000];
const PATIENT_STEPS_MS = [15_000, 30_000, 60_000];

/** `attempt` counts the closes in a row since the last connection that worked, from 0. */
export function planAfterClose(code: number, attempt: number, random: () => number = Math.random): Plan {
  switch (code) {
    case CLOSE.fellBehind:
      return { kind: "retry", delayMs: 0 }; // the server asks for it: everything missed is stored
    case CLOSE.idle:
      return { kind: "pause" };
    case CLOSE.sessionExpired:
      return { kind: "restart" };
    case CLOSE.policy:
      return { kind: "stop", why: "The server does not accept connections from this page." };
    case CLOSE.tooManyConversations:
      return { kind: "stop", why: "This address has started too many conversations for now. Try again in a while." };
    case CLOSE.tooManyConnections:
      return { kind: "stop", why: "Too many tabs are connected. Close one and try again." };
    case CLOSE.atCapacity:
      if (attempt >= PATIENT_STEPS_MS.length) return { kind: "stop", why: "The demo is full right now. Try again in a few minutes." };
      return { kind: "retry", delayMs: jittered(PATIENT_STEPS_MS[attempt], random) };
    default:
      // A dropped network, a restart of the server, a proxy that gave up. Back off, and keep trying.
      return { kind: "retry", delayMs: jittered(QUICK_STEPS_MS[Math.min(attempt, QUICK_STEPS_MS.length - 1)], random) };
  }
}

/** Between half and all of the step, so many tabs that lost one server do not return as one wave. */
function jittered(stepMs: number, random: () => number): number {
  return Math.round(stepMs * (0.5 + 0.5 * random()));
}
