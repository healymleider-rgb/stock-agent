import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "StockEval — Institutional Research Platform",
  description:
    "Institutional-grade equity research powered by AI agents. Deep fundamental, technical, and macro analysis in seconds.",
  keywords: ["stock analysis", "equity research", "AI", "investment", "fundamental analysis"],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className={`${inter.className} antialiased bg-white text-slate-900 min-h-screen`}>
        {children}
      </body>
    </html>
  );
}
