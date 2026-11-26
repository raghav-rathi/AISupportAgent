import type { ReactNode } from "react";

/**
 * Renders the assistant's text: paragraphs, bullet lists, **emphasis** and citation marks such
 * as [2] or [1, 2]. The model writes a small subset of Markdown, and this covers it. Everything goes
 * through React as text, so nothing the model writes can become markup.
 */

const CITATION = /\[\d+(?:\s*,\s*\d+)*\]/;
const INLINE = new RegExp(`(\\*\\*[^*\\n]+\\*\\*|${CITATION.source})`, "g");
const ONLY_CITATION = new RegExp(`^${CITATION.source}$`);
const BULLET = /^\s*[-*]\s+/;

function inline(text: string, keyPrefix: string): ReactNode[] {
  return text.split(INLINE).map((part, i) => {
    const key = `${keyPrefix}-${i}`;
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      return (
        <strong key={key} className="font-w480">
          {part.slice(2, -2)}
        </strong>
      );
    }
    if (ONLY_CITATION.test(part)) {
      return (
        <sup key={key} className="ml-2 text-caption text-ash-text">
          {part.slice(1, -1)}
        </sup>
      );
    }
    return part;
  });
}

interface Block {
  list: boolean;
  lines: string[];
}

/** Runs of bullet lines become lists, runs of other lines paragraphs, and a blank line ends a run. */
function blocks(text: string): Block[] {
  const found: Block[] = [];
  let open: Block | null = null;
  for (const line of text.trim().split("\n")) {
    if (!line.trim()) {
      open = null;
      continue;
    }
    const list = BULLET.test(line);
    if (!open || open.list !== list) {
      open = { list, lines: [] };
      found.push(open);
    }
    open.lines.push(list ? line.replace(BULLET, "") : line);
  }
  return found;
}

export function RichText({ text, caret = false }: { text: string; caret?: boolean }) {
  const all = blocks(text);
  return (
    <div className="flex flex-col gap-12 text-body">
      {all.map((block, b) => {
        const tail =
          caret && b === all.length - 1 ? (
            <span className="ml-2 inline-block h-16 w-2 animate-caret bg-ivory-text align-[-2px]" />
          ) : null;
        if (block.list) {
          return (
            <ul key={b} className="flex list-disc flex-col gap-4 pl-20 marker:text-ash-text">
              {block.lines.map((line, l) => (
                <li key={l}>
                  {inline(line, `${b}-${l}`)}
                  {l === block.lines.length - 1 ? tail : null}
                </li>
              ))}
            </ul>
          );
        }
        return (
          <p key={b} className="whitespace-pre-wrap">
            {inline(block.lines.join("\n"), String(b))}
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
    for (const number of mark[0].matchAll(/\d+/g)) seen.add(Number(number[0]));
  }
  return [...seen];
}
