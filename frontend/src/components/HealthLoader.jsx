import { useEffect, useState } from "react";
import { getHealth } from "../api/client.js";

const MAX_WAIT_MS = 90000; // safety cap so the splash can never trap the user
const MIN_SHOW_MS = 900; // avoid a one-frame flash when the server answers instantly
const SLOW_HINT_MS = 6000;

const css = `
@keyframes mm-needle { 0% { transform: rotate(-35deg); } 20% { transform: rotate(50deg); }
  45% { transform: rotate(-20deg); } 70% { transform: rotate(30deg); } 100% { transform: rotate(-35deg); } }
@keyframes mm-ring { 0% { opacity: .55; transform: scale(.92); } 100% { opacity: 0; transform: scale(1.35); } }
@keyframes mm-fade { from { opacity: 1; } to { opacity: 0; } }
.mm-loader { position: fixed; inset: 0; z-index: 9999; display: flex; flex-direction: column; align-items: center;
  justify-content: center; gap: 14px; padding: 24px; text-align: center; background: var(--bg, #eceee6); color: var(--ink, #1e2620); }
.mm-loader.leaving { animation: mm-fade .35s ease forwards; pointer-events: none; }
.mm-compass { position: relative; width: 120px; height: 120px; }
.mm-compass .ring { position: absolute; inset: 0; border-radius: 50%; border: 2px solid var(--accent, #3f6f63);
  animation: mm-ring 2s ease-out infinite; }
.mm-compass .ring.two { animation-delay: 1s; }
.mm-compass svg { position: relative; width: 100%; height: 100%; }
.mm-needle { transform-origin: 60px 60px; animation: mm-needle 2.6s ease-in-out infinite; }
.mm-loader h1 { margin: 6px 0 0; font-size: clamp(22px, 5vw, 32px); letter-spacing: .14em; font-weight: 700; }
.mm-loader .sub { margin: 0; font-size: 13px; letter-spacing: .32em; color: var(--accent, #3f6f63); font-weight: 600; }
.mm-loader .msg { margin: 8px 0 0; font-size: 13px; color: var(--ink-soft, #576056); min-height: 18px; }
@media (prefers-reduced-motion: reduce) { .mm-needle, .mm-compass .ring { animation: none; } }
`;

// Full-screen splash shown from app start until the first /health check
// settles (ready, warming, failed or unreachable). After that the usual
// ApiStatusBanner takes over for any ongoing status.
export default function HealthLoader() {
  const [phase, setPhase] = useState("loading"); // loading | leaving | done
  const [slow, setSlow] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const started = Date.now();
    const slowTimer = setTimeout(() => !cancelled && setSlow(true), SLOW_HINT_MS);
    let leaveTimer;
    let doneTimer;

    function finish() {
      if (cancelled) return;
      const wait = Math.max(0, MIN_SHOW_MS - (Date.now() - started));
      leaveTimer = setTimeout(() => {
        if (cancelled) return;
        setPhase("leaving");
        doneTimer = setTimeout(() => !cancelled && setPhase("done"), 350);
      }, wait);
    }

    const capTimer = setTimeout(finish, MAX_WAIT_MS);
    getHealth().then(finish, finish);

    return () => {
      cancelled = true;
      clearTimeout(slowTimer);
      clearTimeout(capTimer);
      clearTimeout(leaveTimer);
      clearTimeout(doneTimer);
    };
  }, []);

  if (phase === "done") return null;

  return (
    <div className={`mm-loader${phase === "leaving" ? " leaving" : ""}`} role="status" aria-live="polite">
      <style>{css}</style>
      <div className="mm-compass" aria-hidden="true">
        <span className="ring" />
        <span className="ring two" />
        <svg viewBox="0 0 120 120">
          <circle cx="60" cy="60" r="54" fill="none" stroke="var(--accent, #3f6f63)" strokeWidth="3" />
          <circle cx="60" cy="60" r="46" fill="none" stroke="var(--border, #d4d2c1)" strokeWidth="1" />
          {[0, 90, 180, 270].map((deg) => (
            <line key={deg} x1="60" y1="8" x2="60" y2="18" stroke="var(--ink-soft, #576056)" strokeWidth="2.5"
              transform={`rotate(${deg} 60 60)`} />
          ))}
          <g className="mm-needle">
            <polygon points="60,16 69,60 51,60" fill="var(--brick, #a85742)" />
            <polygon points="60,104 69,60 51,60" fill="var(--ink-faint, #8a9086)" />
          </g>
          <circle cx="60" cy="60" r="5" fill="var(--ink, #1e2620)" />
        </svg>
      </div>
      <h1>MOOD MENTOR</h1>
      <p className="sub">THE MIND COMPASS</p>
      <p className="msg">
        {slow ? "Waking up the server — this can take a little while on first load…" : "Checking server health…"}
      </p>
    </div>
  );
}
