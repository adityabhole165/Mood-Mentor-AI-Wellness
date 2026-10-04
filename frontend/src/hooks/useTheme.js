import { useCallback, useEffect, useRef, useState } from "react";

const KEY = "moodmentor.theme";
const DURATION = 1200; // ms — total sweep of the flowing shape
const STAGGER = 120; // ms — second ribbon follows the first

// Colours the flowing ribbons are painted with, per *target* theme.
const FLOW_COLORS = {
  light: { lead: "#d9e5df", body: "#eceee6" },
  dark: { lead: "#24382f", body: "#121814" },
};

// Wavy edge of the ribbon (leading edge). The trailing edge is the same shape mirrored.
const WAVE_SVG =
  '<svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">' +
  '<path d="M100 0 L100 100 L38 100 C70 88 6 77 38 64 C70 51 6 40 38 28 C70 16 6 8 38 0 Z" fill="currentColor"/>' +
  "</svg>";

function initialTheme() {
  try {
    const saved = localStorage.getItem(KEY);
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    // storage unavailable (private mode) — fall through
  }
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function buildSheet(color, delay) {
  const sheet = document.createElement("div");
  sheet.className = "theme-flow-sheet";
  sheet.style.color = color;
  sheet.style.animationDelay = `${delay}ms`;
  sheet.innerHTML =
    `<span class="theme-flow-wave">${WAVE_SVG}</span>` +
    '<span class="theme-flow-fill"></span>' +
    `<span class="theme-flow-wave theme-flow-wave-end">${WAVE_SVG}</span>`;
  return sheet;
}

export default function useTheme() {
  const [theme, setTheme] = useState(initialTheme);
  const busy = useRef(false);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem(KEY, theme);
    } catch {
      // ignore
    }
  }, [theme]);

  const toggle = useCallback(() => {
    if (busy.current) return;
    const next = theme === "dark" ? "light" : "dark";
    const root = document.documentElement;
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

    if (reduce) {
      setTheme(next);
      return;
    }

    busy.current = true;
    const colors = FLOW_COLORS[next];

    // Flowing shape: two ribbons sweeping left → right across the screen.
    const overlay = document.createElement("div");
    overlay.className = "theme-flow";
    overlay.setAttribute("aria-hidden", "true");
    overlay.style.setProperty("--flow-ms", `${DURATION}ms`);
    overlay.appendChild(buildSheet(colors.lead, 0));
    overlay.appendChild(buildSheet(colors.body, STAGGER));
    document.body.appendChild(overlay);

    // Smoothly fade every colour while the screen is covered.
    root.classList.add("theme-switching");

    // Screen is fully covered at the midpoint of the second ribbon.
    const swapAt = STAGGER + DURATION / 2;
    window.setTimeout(() => setTheme(next), swapAt);

    window.setTimeout(() => {
      overlay.remove();
      root.classList.remove("theme-switching");
      busy.current = false;
    }, STAGGER + DURATION + 150);
  }, [theme]);

  return [theme, toggle];
}
