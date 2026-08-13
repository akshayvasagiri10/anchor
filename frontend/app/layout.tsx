import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Anchor — grounded document Q&A",
  description:
    "A hybrid-retrieval RAG chatbot. Every answer cites the source it came from.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
