import type { NextConfig } from "next";

/**
 * The backend is the only other origin this page talks to. Its address is known at build time,
 * so the policy can name it. Without one, as in local development, any WebSocket is allowed.
 */
function backendOrigin(): string {
  const url = process.env.NEXT_PUBLIC_WS_URL;
  if (!url) return "ws: wss:";
  try {
    return new URL(url).origin;
  } catch {
    return "ws: wss:";
  }
}

const development = process.env.NODE_ENV !== "production";

// Next.js writes inline scripts into every page to hand the rendered tree to React, so scripts
// cannot be limited to files without a per-request nonce, which a statically served page has
// no request to make one in. What the policy does do: nothing is loaded from another origin,
// the page talks to no server but its own backend, and it cannot be put in a frame.
const contentSecurityPolicy = [
  "default-src 'self'",
  `script-src 'self' 'unsafe-inline'${development ? " 'unsafe-eval'" : ""}`,
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data:",
  "font-src 'self'",
  `connect-src 'self' ${backendOrigin()}`,
  "object-src 'none'",
  "base-uri 'self'",
  "form-action 'self'",
  "frame-ancestors 'none'",
].join("; ");

const securityHeaders = [
  { key: "Content-Security-Policy", value: contentSecurityPolicy },
  { key: "X-Frame-Options", value: "DENY" },
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
  // The microphone is for voice mode, on this page only. Nothing else is asked for.
  { key: "Permissions-Policy", value: "camera=(), geolocation=(), microphone=(self)" },
];

const nextConfig: NextConfig = {
  cacheComponents: true,
  partialPrefetching: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
  async headers() {
    return [{ source: "/:path*", headers: securityHeaders }];
  },
  async rewrites() {
    // Browsers ask for this path whatever the page says its icon is.
    return [{ source: "/favicon.ico", destination: "/icon.svg" }];
  },
};

export default nextConfig;
