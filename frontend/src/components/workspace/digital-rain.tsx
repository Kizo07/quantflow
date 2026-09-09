"use client";

import { PauseIcon, PlayIcon } from "lucide-react";
import { useEffect, useRef, useState } from "react";

// Digital rain ported from the Kizo07 portfolio site (assets/matrix.js):
// cyan streams with occasional ledger-gold ones, rendered on one
// viewport-sized decorative canvas over the workspace app. The canvas is
// pointer-events-none and sits below dialogs/menus (z-50) so the app stays
// fully interactive; a horizontal mask keeps the busy center calm.
const GLYPHS =
  "01アイウエオカキクケコサシスセソタチツテトナニヌネノラリルレロΣλΔΩπ";
const MOTION_STORAGE_KEY = "quantflow-motion";

type Stream = {
  x: number;
  start: number;
  length: number;
  speed: number;
  brightness: number;
  warm: boolean;
};

// Stable initial composition: immediately visible, including without motion.
function noise(value: number) {
  const n = Math.sin(value * 127.1 + 311.7) * 43758.5453;
  return n - Math.floor(n);
}

export function DigitalRain() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const toggleRef = useRef<HTMLButtonElement | null>(null);
  const pausedRef = useRef(false);
  const [paused, setPaused] = useState(false);
  const [reducedMotion, setReducedMotion] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    const root = document.documentElement;
    const reducedMotionQuery = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    );

    try {
      pausedRef.current = localStorage.getItem(MOTION_STORAGE_KEY) === "paused";
    } catch {}

    let width = 0;
    let height = 0;
    let streams: Stream[] = [];
    let elapsed = 0;
    let lastFrame = 0;
    let frame = 0;
    let light = root.classList.contains("light");
    let font = "12px monospace";

    const draw = () => {
      context.clearRect(0, 0, width, height);
      context.font = font;
      context.textAlign = "center";
      streams.forEach((stream, column) => {
        const cycle = height + stream.length * 20;
        const head = (stream.start + elapsed * stream.speed) % cycle;
        for (let row = 0; row < stream.length; row++) {
          const y = head - row * 20;
          if (y < -20 || y > height + 20) continue;
          const alpha =
            Math.pow(1 - row / stream.length, 1.4) * stream.brightness;
          if (stream.warm) {
            // Occasional ledger-gold stream: the warm counterpoint in the rain.
            context.fillStyle = light
              ? `rgba(138,94,22,${alpha})`
              : `rgba(236,182,96,${alpha})`;
          } else {
            context.fillStyle = light
              ? `rgba(0,84,132,${alpha})`
              : `rgba(0,178,255,${alpha})`;
          }
          if (row === 0) {
            if (stream.warm) {
              context.fillStyle = light ? "#7d5518" : "#ffdda6";
              context.shadowColor = "#e3ac55";
            } else {
              context.fillStyle = light ? "#005f88" : "#a2e9ff";
              context.shadowColor = "#00bfff";
            }
            context.shadowBlur = light ? 0 : 10;
          }
          const index = Math.floor(
            noise(column * 41 + row + Math.floor(elapsed * 0.7)) *
              GLYPHS.length,
          );
          context.fillText(GLYPHS.charAt(index), stream.x, y);
          if (row === 0) context.shadowBlur = 0;
        }
      });
    };

    const tick = (timestamp: number) => {
      if (!lastFrame) lastFrame = timestamp;
      const delta = timestamp - lastFrame;
      // Cap drawing at 20fps and avoid catch-up work after a suspended tab.
      if (delta >= 50) {
        elapsed += Math.min(delta, 100) / 1000;
        lastFrame = timestamp;
        draw();
      }
      frame = requestAnimationFrame(tick);
    };

    const resize = () => {
      // The fixed canvas stays viewport-sized even on long pages.
      const nextWidth = canvas.clientWidth;
      const nextHeight = canvas.clientHeight;
      if (nextWidth === width && nextHeight === height) return;
      width = nextWidth;
      height = nextHeight;
      const plex = getComputedStyle(root)
        .getPropertyValue("--font-plex-mono")
        .trim();
      font = plex ? `12px ${plex}, monospace` : "12px monospace";
      const ratio = Math.min(window.devicePixelRatio || 1, 1.5);
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      streams = [];
      const spacing = width < 640 ? 26 : 23;
      for (let i = 0; i < Math.ceil(width / spacing); i++) {
        streams.push({
          x: i * spacing + 8,
          start: noise(i + 1) * (height + 480),
          length: 14 + Math.floor(noise(i + 31) * 22),
          speed: 18 + noise(i + 73) * 28,
          brightness: 0.18 + noise(i + 111) * 0.6,
          warm: noise(i + 151) < 0.14,
        });
      }
      draw();
    };

    const sync = () => {
      setReducedMotion(reducedMotionQuery.matches);
      setPaused(pausedRef.current);
      cancelAnimationFrame(frame);
      frame = 0;
      lastFrame = 0;
      if (
        !pausedRef.current &&
        !reducedMotionQuery.matches &&
        !document.hidden
      ) {
        frame = requestAnimationFrame(tick);
      }
    };

    const handleToggle = () => {
      pausedRef.current = !pausedRef.current;
      try {
        localStorage.setItem(
          MOTION_STORAGE_KEY,
          pausedRef.current ? "paused" : "running",
        );
      } catch {}
      sync();
    };

    const themeObserver = new MutationObserver(() => {
      light = root.classList.contains("light");
      draw();
    });
    themeObserver.observe(root, {
      attributes: true,
      attributeFilter: ["class"],
    });

    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(canvas);

    const handleVisibilityChange = () => sync();
    const handlePageHide = () => cancelAnimationFrame(frame);
    const handlePageShow = () => sync();

    const toggle = toggleRef.current;
    toggle?.addEventListener("click", handleToggle);
    reducedMotionQuery.addEventListener("change", sync);
    document.addEventListener("visibilitychange", handleVisibilityChange);
    window.addEventListener("pagehide", handlePageHide);
    window.addEventListener("pageshow", handlePageShow);

    resize();
    sync();

    return () => {
      toggle?.removeEventListener("click", handleToggle);
      reducedMotionQuery.removeEventListener("change", sync);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      window.removeEventListener("pagehide", handlePageHide);
      window.removeEventListener("pageshow", handlePageShow);
      themeObserver.disconnect();
      resizeObserver.disconnect();
      cancelAnimationFrame(frame);
    };
  }, []);

  const label = paused
    ? "Play background animation"
    : "Pause background animation";

  return (
    <>
      <canvas
        ref={canvasRef}
        aria-hidden="true"
        className="pointer-events-none fixed inset-0 z-30 h-full w-full [mask-image:linear-gradient(90deg,#000,rgba(0,0,0,0.5)_25%,rgba(0,0,0,0.3)_50%,rgba(0,0,0,0.5)_75%,#000)] opacity-30"
      />
      <button
        ref={toggleRef}
        type="button"
        hidden={reducedMotion}
        aria-label={label}
        title={label}
        className="bg-background/80 text-muted-foreground border-border hover:border-foreground/40 hover:text-foreground fixed right-6 bottom-6 z-40 grid size-8 place-items-center rounded-md border backdrop-blur-sm transition-colors"
      >
        {paused ? (
          <PlayIcon className="size-3.5" aria-hidden="true" />
        ) : (
          <PauseIcon className="size-3.5" aria-hidden="true" />
        )}
      </button>
    </>
  );
}
