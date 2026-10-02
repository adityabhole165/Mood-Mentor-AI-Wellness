import { useEffect, useMemo, useState } from "react";
import { getEmotionHistory } from "../api/client.js";
import { EMOTIONS, clamp01, emotionColor, emotionEmoji, emotionLabel, hexToRgba } from "../utils/emotions.js";

const WEEKS = 26;
const DAY_MS = 86400000;

function dayKey(d) {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

// Collapse every entry on a given day into one pixel: the most frequent
// dominant emotion wins, tinted by that day's average intensity.
function aggregateByDay(items) {
  const days = new Map();
  for (const item of items) {
    const d = new Date(item.timestamp);
    if (Number.isNaN(d.getTime()) || !item.dominant_emotion) continue;
    const key = dayKey(d);
    const entry = days.get(key) || { counts: {}, intensitySum: 0, n: 0 };
    entry.counts[item.dominant_emotion] = (entry.counts[item.dominant_emotion] || 0) + 1;
    entry.intensitySum += clamp01(item.intensity);
    entry.n += 1;
    days.set(key, entry);
  }
  const out = new Map();
  for (const [key, e] of days) {
    const [emotion] = Object.entries(e.counts).sort((a, b) => b[1] - a[1])[0];
    out.set(key, { emotion, intensity: e.intensitySum / e.n, entries: e.n });
  }
  return out;
}

export default function YearInPixels({ userId, refreshKey = 0 }) {
  const [state, setState] = useState({ status: "loading", items: [] });
  const [selected, setSelected] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setState({ status: "loading", items: [] });
    getEmotionHistory(userId)
      .then((res) => {
        if (cancelled) return;
        if (!res.available) setState({ status: "unavailable", items: [] });
        else setState({ status: "ready", items: res.data?.items || [] });
      })
      .catch((err) => !cancelled && setState({ status: "error", items: [], message: err.message }));
    return () => {
      cancelled = true;
    };
  }, [userId, refreshKey]);

  const byDay = useMemo(() => aggregateByDay(state.items), [state.items]);

  // Grid: columns = weeks (oldest → newest), rows = Mon…Sun.
  const columns = useMemo(() => {
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const mondayOffset = (today.getDay() + 6) % 7;
    const start = new Date(today.getTime() - (mondayOffset + (WEEKS - 1) * 7) * DAY_MS);
    return Array.from({ length: WEEKS }, (_, w) =>
      Array.from({ length: 7 }, (_, d) => {
        const date = new Date(start.getTime() + (w * 7 + d) * DAY_MS);
        return { date, key: dayKey(date), future: date > today };
      })
    );
  }, []);

  const usedEmotions = useMemo(() => {
    const set = new Set();
    for (const v of byDay.values()) set.add(v.emotion);
    return EMOTIONS.filter((e) => set.has(e));
  }, [byDay]);

  const selectedInfo = selected ? byDay.get(selected) : null;

  return (
    <div className="panel">
      <h2 style={{ fontSize: 18 }}>Year in pixels</h2>
      <p className="hint" style={{ margin: "4px 0 14px" }}>
        One square per day, coloured by that day's most frequent emotion — deeper colour means more intense.
      </p>

      {state.status === "loading" ? <p className="hint">Loading your days…</p> : null}
      {state.status === "unavailable" ? <p className="not-available">Not available on this backend.</p> : null}
      {state.status === "error" ? <div className="error-banner" role="alert">{state.message}</div> : null}

      {state.status === "ready" ? (
        <>
          {byDay.size === 0 ? (
            <p className="hint">No check-ins yet — your first one will colour today's square.</p>
          ) : null}
          <div className="pixels-scroll">
            <div className="pixels" role="grid" aria-label="Daily dominant emotion, last 26 weeks">
              {columns.map((col, w) => (
                <div className="pixel-col" role="row" key={w}>
                  {col.map((cell) => {
                    const info = byDay.get(cell.key);
                    const label = cell.future
                      ? ""
                      : info
                      ? `${cell.date.toDateString()}: ${emotionLabel(info.emotion)}, intensity ${Math.round(info.intensity * 100)}%, ${info.entries} check-in${info.entries > 1 ? "s" : ""}`
                      : `${cell.date.toDateString()}: no check-in`;
                    return (
                      <button
                        type="button"
                        role="gridcell"
                        key={cell.key}
                        className={`pixel${cell.future ? " pixel-future" : ""}${selected === cell.key ? " pixel-selected" : ""}`}
                        style={
                          info
                            ? { background: hexToRgba(emotionColor(info.emotion), 0.35 + 0.65 * info.intensity) }
                            : undefined
                        }
                        title={label}
                        aria-label={label || undefined}
                        disabled={cell.future}
                        onClick={() => setSelected(cell.key)}
                      />
                    );
                  })}
                </div>
              ))}
            </div>
          </div>

          <div className="pixel-legend">
            {(usedEmotions.length ? usedEmotions : EMOTIONS).map((e) => (
              <span key={e} className="pill">
                <span className="legend-dot" style={{ background: emotionColor(e) }} aria-hidden="true" />
                {emotionEmoji(e)} {emotionLabel(e)}
              </span>
            ))}
          </div>

          {selected ? (
            <p className="hint" style={{ marginTop: 10 }} aria-live="polite">
              {selectedInfo
                ? `${new Date(selected + "T00:00:00").toDateString()} — mostly ${emotionLabel(selectedInfo.emotion)} (${Math.round(
                    selectedInfo.intensity * 100
                  )}% intensity, ${selectedInfo.entries} check-in${selectedInfo.entries > 1 ? "s" : ""}).`
                : `${new Date(selected + "T00:00:00").toDateString()} — no check-in.`}
            </p>
          ) : null}
        </>
      ) : null}
    </div>
  );
}
