import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

// Both are variable fonts, so the design's in-between weight, 450, renders as drawn.
const sans = Geist({ subsets: ["latin"], variable: "--font-geist-sans", display: "swap" });
const mono = Geist_Mono({ subsets: ["latin"], variable: "--font-geist-mono", display: "swap" });

export const metadata: Metadata = {
  title: "Fernhill Support",
  description:
    "A support assistant for Fernhill, an invented company. It answers from the help center, checks who it is talking to, shows every step it takes, and hands what it cannot solve to a person at the service desk.",
};

export const viewport: Viewport = { themeColor: "#fafafa" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable} h-full antialiased`}>
      <body className="h-full">{children}</body>
    </html>
  );
}
