import { describe, expect, it } from "vitest";

import { CLOSE, planAfterClose } from "./reconnect";

const noJitter = () => 1;

describe("after the socket closes", () => {
  it("comes straight back when the server says the client only fell behind", () => {
    expect(planAfterClose(CLOSE.fellBehind, 3, noJitter)).toEqual({ kind: "retry", delayMs: 0 });
  });

  it("backs off after a dropped network, and keeps trying", () => {
    const delays = [0, 1, 2, 3, 4, 50].map((attempt) => planAfterClose(1006, attempt, noJitter));
    expect(delays.map((plan) => (plan.kind === "retry" ? plan.delayMs : -1))).toEqual([500, 1000, 2000, 4000, 8000, 8000]);
  });

  it("spreads its retries, so many tabs do not return as one wave", () => {
    const low = planAfterClose(1006, 2, () => 0);
    const high = planAfterClose(1006, 2, () => 1);
    expect([low, high]).toEqual([
      { kind: "retry", delayMs: 1000 },
      { kind: "retry", delayMs: 2000 },
    ]);
  });

  it("waits for the visitor after an idle close", () => {
    expect(planAfterClose(CLOSE.idle, 0)).toEqual({ kind: "pause" });
  });

  it("does not knock again where knocking cannot help", () => {
    for (const code of [CLOSE.policy, CLOSE.tooManyConversations, CLOSE.tooManyConnections]) {
      expect(planAfterClose(code, 0).kind).toBe("stop");
    }
  });

  it("gives a full server a few patient tries and then stops", () => {
    const plans = [0, 1, 2, 3].map((attempt) => planAfterClose(CLOSE.atCapacity, attempt, noJitter));
    expect(plans.slice(0, 3)).toEqual([
      { kind: "retry", delayMs: 15000 },
      { kind: "retry", delayMs: 30000 },
      { kind: "retry", delayMs: 60000 },
    ]);
    expect(plans[3].kind).toBe("stop");
  });

  it("starts a new conversation when the old one is gone", () => {
    expect(planAfterClose(CLOSE.sessionExpired, 0)).toEqual({ kind: "restart" });
  });
});
