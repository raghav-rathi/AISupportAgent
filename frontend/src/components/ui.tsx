import type { ButtonHTMLAttributes, HTMLAttributes, ReactNode, Ref } from "react";

/**
 * The few controls the design allows. Corners are 6px. The filled obsidian button is the one
 * primary action of a view. Everything else is a hairline or nothing. Labels are mono stamps.
 */

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "ghost" | "quiet" | "surface";
  size?: "md" | "sm";
  ref?: Ref<HTMLButtonElement>;
};

const BUTTON_BASE =
  "inline-flex shrink-0 items-center justify-center gap-8 whitespace-nowrap rounded-md font-sans font-regular " +
  "transition-colors disabled:cursor-not-allowed disabled:opacity-40";

const BUTTON_SIZES = {
  md: "h-36 px-12 text-body-sm",
  sm: "h-28 px-8 text-caption",
};

const BUTTON_VARIANTS = {
  primary: "bg-obsidian text-pure-white hover:bg-charcoal disabled:hover:bg-obsidian",
  ghost: "bg-pure-white text-obsidian shadow-hairline hover:bg-paper-white hover:shadow-ring",
  quiet: "text-charcoal hover:bg-hairline/70 hover:text-obsidian",
  surface: "bg-hairline text-obsidian hover:bg-ash/60",
};

export function Button({ variant = "ghost", size = "md", className = "", type = "button", ...rest }: ButtonProps) {
  return <button type={type} className={`${BUTTON_BASE} ${BUTTON_SIZES[size]} ${BUTTON_VARIANTS[variant]} ${className}`} {...rest} />;
}

/** A pill with a mono stamp in it: a state, a plan, a priority. */
export function Tag({ children, className = "", tone = "plain" }: { children: ReactNode; className?: string; tone?: "plain" | "solid" | "good" }) {
  const tones = {
    plain: "bg-pure-white text-charcoal shadow-hairline",
    solid: "bg-obsidian text-pure-white",
    good: "bg-pure-white text-terminal-green shadow-hairline",
  };
  return <span className={`label-mono inline-flex h-20 shrink-0 items-center whitespace-nowrap rounded-full px-8 ${tones[tone]} ${className}`}>{children}</span>;
}

/** A small label above a block: the mono stamp, in the muted tone. */
export function Caption({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <p className={`label-mono text-stone ${className}`}>{children}</p>;
}

/** The label that announces a section, in full strength. */
export function Eyebrow({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <p className={`label-mono text-obsidian ${className}`}>{children}</p>;
}

/** A white surface one step above the page, held by a hairline ring and never by a shadow. */
export function Card({ className = "", ...rest }: HTMLAttributes<HTMLDivElement>) {
  return <div className={`rounded-md bg-pure-white shadow-card ${className}`} {...rest} />;
}

const GLYPHS = {
  check: <path d="M3.5 8.5l3 3 6-7" />,
  cross: <path d="M4 4l8 8M12 4l-8 8" />,
  minus: <path d="M4 8h8" />,
  clock: (
    <>
      <circle cx="8" cy="8" r="5.5" />
      <path d="M8 5v3.2l2 1.3" />
    </>
  ),
  swap: <path d="M3 5.5h9l-2.2-2.2M13 10.5H4l2.2 2.2" />,
  shield: <path d="M8 2.2l4.8 1.8v3.6c0 2.9-2 5.2-4.8 6.2-2.8-1-4.8-3.3-4.8-6.2V4z" />,
  person: (
    <>
      <circle cx="8" cy="5.5" r="2.5" />
      <path d="M3 13.5c.6-2.4 2.6-3.6 5-3.6s4.4 1.2 5 3.6" />
    </>
  ),
  panel: (
    <>
      <rect x="2" y="3" width="12" height="10" rx="2" />
      <path d="M10 3v10" />
    </>
  ),
  plus: <path d="M8 3.5v9M3.5 8h9" />,
  arrow: <path d="M8 12.5v-9M4.2 7.2L8 3.4l3.8 3.8" />,
  stop: <rect x="4.5" y="4.5" width="7" height="7" rx="1.5" />,
  left: <path d="M9.5 3.5L5 8l4.5 4.5" />,
  right: <path d="M6.5 3.5L11 8l-4.5 4.5" />,
  alert: (
    <>
      <path d="M8 2.5l6 10.5H2z" />
      <path d="M8 6.5v3M8 11.3v.2" />
    </>
  ),
  mic: (
    <>
      <rect x="6" y="2" width="4" height="7.5" rx="2" />
      <path d="M3.5 7.5a4.5 4.5 0 0 0 9 0M8 12v2" />
    </>
  ),
  inbox: <path d="M2.5 9.5l1.8-6h7.4l1.8 6v3h-11zM2.5 9.5h3.2l.8 1.5h3l.8-1.5h3.2" />,
  chat: <path d="M2.5 3.5h11v7.5H7l-3 2.5v-2.5H2.5z" />,
  note: (
    <>
      <path d="M3.5 2.5h6.5l2.5 2.5v8.5h-9z" />
      <path d="M5.8 7h4.4M5.8 9.5h4.4" />
    </>
  ),
  reply: <path d="M6.5 4L3 7.5 6.5 11M3.2 7.5h6.3a3.5 3.5 0 0 1 3.5 3.5v1" />,
  search: (
    <>
      <circle cx="7" cy="7" r="4.2" />
      <path d="M10.2 10.2l3.3 3.3" />
    </>
  ),
  card: (
    <>
      <rect x="2" y="3.5" width="12" height="9" rx="1.5" />
      <path d="M2 6.5h12" />
    </>
  ),
  dot: <circle cx="8" cy="8" r="2.5" fill="currentColor" stroke="none" />,
  down: <path d="M3.5 6.5L8 11l4.5-4.5" />,
} as const;

export type GlyphName = keyof typeof GLYPHS;

/** A 16px line glyph that takes the text colour around it. */
export function Glyph({ name, className = "" }: { name: GlyphName; className?: string }) {
  return (
    <svg
      viewBox="0 0 16 16"
      width="16"
      height="16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.4"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className={`shrink-0 ${className}`}
    >
      {GLYPHS[name]}
    </svg>
  );
}

export function Spinner({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 16 16" width="16" height="16" fill="none" aria-hidden="true" className={`shrink-0 animate-spin-slow ${className}`}>
      <circle cx="8" cy="8" r="5.5" stroke="currentColor" strokeOpacity="0.25" strokeWidth="1.4" />
      <path d="M13.5 8A5.5 5.5 0 0 0 8 2.5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  );
}
