import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

// The variable font, so the design's in-between weights (360, 420, 480, 530) render as drawn.
const inter = Inter({ subsets: ["latin"], variable: "--font-inter", display: "swap" });

export const metadata: Metadata = {
  title: "Fernhill Support",
  description:
    "A support assistant for Fernhill, an invented company. It answers from the help center, checks who it is talking to, and shows every step it takes.",
};

export const viewport: Viewport = { themeColor: "#171721" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} h-full antialiased`}>
      <body className="h-full">{children}</body>
    </html>
  );
}
