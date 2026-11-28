/**
 * The workspace of this browser, and the name its service desk acts under.
 *
 * A workspace is the conversations one browser has started. The browser makes up a key once and
 * keeps it. Every conversation it opens joins that workspace, and the desk shows the tickets of
 * that workspace and of no other. There is no login in this demo: the key is what a login
 * would be.
 */

const WORKSPACE = "fernhill.workspace";
const AGENT = "fernhill.agent";
export const DEFAULT_AGENT = "Sam Carter";

let memory: string | null = null; // when storage is blocked, the workspace lives as long as the page

function fresh(): string {
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

export function workspaceKey(): string {
  try {
    const kept = window.localStorage.getItem(WORKSPACE);
    if (kept && kept.length >= 16) return kept;
    const made = fresh();
    window.localStorage.setItem(WORKSPACE, made);
    return made;
  } catch {
    memory ??= fresh();
    return memory;
  }
}

export function agentName(): string {
  try {
    return window.localStorage.getItem(AGENT)?.trim() || DEFAULT_AGENT;
  } catch {
    return DEFAULT_AGENT;
  }
}

export function rememberAgent(name: string): void {
  try {
    window.localStorage.setItem(AGENT, name);
  } catch {
    // see workspaceKey()
  }
}

/** The desk's socket sits next to the conversation's, on the same backend. */
export function deskUrl(conversationUrl: string): string {
  return conversationUrl.replace(/\/ws\/?$/, "/desk");
}
