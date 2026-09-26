"use client";

import { usePathname } from "next/navigation";
import { ThemeProvider as NextThemesProvider, useTheme } from "next-themes";
import { useEffect } from "react";

const THEME_COLORS = { dark: "#020609", light: "#f5faff" } as const;

/**
 * Compat shim for the website theme contract: mirror the resolved next-themes
 * mode onto `document.documentElement.dataset.theme` (for tearsheet-iframe
 * CSS injection) and keep `<meta name="theme-color">` in sync.
 */
function ThemeCompat() {
  const { resolvedTheme } = useTheme();

  useEffect(() => {
    const theme = resolvedTheme === "light" ? "light" : "dark";
    document.documentElement.dataset.theme = theme;
    let meta = document.querySelector('meta[name="theme-color"]');
    if (!meta) {
      meta = document.createElement("meta");
      meta.setAttribute("name", "theme-color");
      document.head.appendChild(meta);
    }
    meta.setAttribute("content", THEME_COLORS[theme]);
  }, [resolvedTheme]);

  return null;
}

export function ThemeProvider({
  children,
  ...props
}: React.ComponentProps<typeof NextThemesProvider>) {
  const pathname = usePathname();
  return (
    <NextThemesProvider
      {...props}
      forcedTheme={pathname === "/" ? "dark" : undefined}
    >
      <ThemeCompat />
      {children}
    </NextThemesProvider>
  );
}
