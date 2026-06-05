import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VidCompare — RAG chat for two videos",
  description: "Compare two social videos with a RAG chat that streams citations.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
