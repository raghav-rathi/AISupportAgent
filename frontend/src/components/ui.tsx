import type { ButtonHTMLAttributes, ReactNode, Ref } from "react";

/**
 * The few controls the design allows. Every control is a pill. Cobalt fills exactly one of them
 * per screen, the primary action. Everything else is ivory on dark.
 */

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "ghost" | "quiet" | "surface";
  ref?: Ref<HTMLButtonElement>;
};

const BUTTON_BASE =
  "inline-flex h-40 shrink-0 items-center justify-center gap-8 whitespace-nowrap px-20 text-body-sm font-regular " +
  "transition-opacity disabled:cursor-not-allowed disabled:opacity-40";

const BUTTON_VARIANTS = {
  primary: "rounded-buttons bg-cobalt text-pure-white hover:opacity-90",
  ghost: "rounded-nav border border-ivory-text text-ivory-text hover:bg-obsidian-button",
  quiet: "rounded-nav text-ivory-text hover:bg-obsidian-button",
  surface: "rounded-buttons bg-obsidian-button text-ivory-text hover:opacity-80",
};

export function Button({ variant = "ghost", className = "", type = "button", ...rest }: ButtonProps) {
  return <button type={type} className={`${BUTTON_BASE} ${BUTTON_VARIANTS[variant]} ${className}`} {...rest} />;
}

export function Tag({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <span
      className={`inline-flex h-24 items-center rounded-tags bg-obsidian-button px-12 text-caption font-w480 text-ash-text ${className}`}
    >
      {children}
    </span>
  );
}

/** A small label above a block, in the muted tone. */
export function Caption({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <p className={`text-caption font-w480 text-ash-text ${className}`}>{children}</p>;
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
