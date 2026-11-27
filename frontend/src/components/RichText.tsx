import type { ReactNode } from "react";

/**
 * Renders the assistant's text: paragraphs, bullet and numbered lists, **emphasis**, `code` and
 * citation marks such as [2], [1, 2] or [1-3]. The model writes a small subset of Markdown, and
 * this covers it. Everything goes through React as text, so nothing the model writes can become
 * markup.
 */

const CITATION = /\[\d+(?:\s*[,\u2013-]\s*\d+)*\]/;
const INLINE = new RegExp(`(\\*\\*[^*\\n]+\\*\\*|\`[^\`\\n]+\`|${CITATION.source})`, "g");
const ONLY_CITATION = new RegExp(`^${CITATION.source}$`);
const BULLET = /^\s*[-*]\s+/;
const NUMBERED = /^\s*\d+[.)]\s+/;
const HEADING = /^\s*#{1,4}\s+/;

/** How citation marks behave: which numbers point at a passage, which is open, and what a click does. */
export interface Cites {
  valid: ReadonlySet<number>;
  open: number | null;
  onCite: (n: number) => void;
}

/** The numbers in one mark: "[1, 3]" gives 1 and 3, "[1-3]" gives 1, 2 and 3. */
function numbersIn(mark: string): number[] {
  const found: number[] = [];
  for (const part of mark.slice(1, -1).split(",")) {
    const [from, to] = part.split(/[\u2013-]/).map((n) => Number(n.trim()));
    if (!Number.isInteger(from)) continue;
    const end = Number.isInteger(to) && to >= from && to - from < 20 ? to : from;
    for (let n = from; n <= end; n += 1) found.push(n);
  }
  return found;
}

function inline(text: string, keyPrefix: string, cites?: Cites): ReactNode[] {
  return text.split(INLINE).map((part, i) => {
    const key = `${keyPrefix}-${i}`;
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      return (
        <strong key={key} className="font-w480">
          {inline(part.slice(2, -2), key, cites)}
        </strong>
      );
    }
    if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
      return (
        <code key={key} className="rounded-md bg-obsidian-button px-4 font-mono text-body-sm">
          {part.slice(1, -1)}
        </code>
      );
    }
    if (ONLY_CITATION.test(part)) {
      const numbers = numbersIn(part);
      return (
        <sup key={key} className="ml-2 text-caption text-ash-text">
          {/* A mark can name a number twice, as in [1, 1], so its place in the mark is the key. */}
          {numbers.map((n, at) => (
            <span key={at}>
              {at > 0 ? ", " : ""}
              {cites?.valid.has(n) ? (
                <button
                  type="button"
                  aria-label={`Source ${n}`}
                  aria-expanded={cites.open === n}
                  onClick={() => cites.onCite(n)}
                  className="-my-8 rounded-md px-4 py-8 underline decoration-slate-border underline-offset-2 hover:text-ivory-text"
                >
                  {n}
                </button>
              ) : (
                n
              )}
            </span>
          ))}
        </sup>
      );
    }
    return part;
  });
}

interface Block {
  kind: "text" | "bullets" | "numbers" | "heading";
  lines: string[];
}

/** Runs of list lines become lists, runs of other lines paragraphs, and a blank line ends a run. */
function blocks(text: string): Block[] {
  const found: Block[] = [];
  let open: Block | null = null;
  for (const line of text.trim().split("\n")) {
    if (!line.trim()) {
      open = null;
      continue;
    }
    const kind = BULLET.test(line) ? "bullets" : NUMBERED.test(line) ? "numbers" : HEADING.test(line) ? "heading" : "text";
    if (!open || open.kind !== kind || kind === "heading") {
      open = { kind, lines: [] };
      found.push(open);
    }
    open.lines.push(line.replace(kind === "bullets" ? BULLET : kind === "numbers" ? NUMBERED : kind === "heading" ? HEADING : /^$/, ""));
  }
  return found;
}

export function RichText({ text, caret = false, cites }: { text: string; caret?: boolean; cites?: Cites }) {
  const all = blocks(text);
  return (
    <div className="flex flex-col gap-12 break-words text-body">
      {all.map((block, b) => {
        const tail =
          caret && b === all.length - 1 ? (
            <span aria-hidden="true" className="ml-2 inline-block h-16 w-2 animate-caret bg-ivory-text align-[-2px]" />
          ) : null;
        if (block.kind === "bullets" || block.kind === "numbers") {
          const List = block.kind === "bullets" ? "ul" : "ol";
          return (
            <List key={b} className={`flex flex-col gap-4 pl-20 marker:text-ash-text ${block.kind === "bullets" ? "list-disc" : "list-decimal"}`}>
              {block.lines.map((line, l) => (
                <li key={l}>
                  {inline(line, `${b}-${l}`, cites)}
                  {l === block.lines.length - 1 ? tail : null}
                </li>
              ))}
            </List>
          );
        }
        return (
          <p key={b} className={`whitespace-pre-wrap ${block.kind === "heading" ? "font-w480" : ""}`}>
            {inline(block.lines.join("\n"), String(b), cites)}
            {tail}
          </p>
        );
      })}
    </div>
  );
}

/** The citation numbers a text uses, in order of first appearance. */
export function citedNumbers(text: string): number[] {
  const seen = new Set<number>();
  for (const mark of text.matchAll(new RegExp(CITATION.source, "g"))) {
    for (const number of numbersIn(mark[0])) seen.add(number);
  }
  return [...seen];
}
