import "@/styles/globals.css";

import { type Metadata } from "next";
import { IBM_Plex_Mono, Inter } from "next/font/google";

import { ThemeProvider } from "@/components/theme-provider";
import { DEFAULT_LOCALE } from "@/core/i18n/locale";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const plexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "700"],
  variable: "--font-plex-mono",
});

export const metadata: Metadata = {
  title: "quantflow",
  description: "A LangChain-based framework for building super agents.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang={DEFAULT_LOCALE}
      suppressContentEditableWarning
      suppressHydrationWarning
    >
      <body className={`${inter.variable} ${plexMono.variable}`}>
        <ThemeProvider attribute="class" enableSystem disableTransitionOnChange>
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}
