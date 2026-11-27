import type { MetadataRoute } from "next";

/** A demo, not a site to index. A crawler that opens the page would only start a conversation. */
export default function robots(): MetadataRoute.Robots {
  return { rules: { userAgent: "*", disallow: "/" } };
}
