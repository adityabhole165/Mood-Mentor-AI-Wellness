import { useMemo, useState } from "react";
import { emotionColor, emotionEmoji, emotionLabel } from "../utils/emotions.js";

// Dependency-free SVG charts for the dashboard:
//   - EmotionTrendChart     : avg intensity / avg polarity per period (line)
//   - EmotionFrequencyChart : how often each emotion occurred (bars)
//   - EmotionScoreChart     : intensity / confidence / polarity score per analysis (line)
// Hover over any point or bar to see its exact value.

const W = 640;
const H = 240;
const PAD = { top: 16, right: 16, bottom: 44, left: 44 };
const PLOT_W = W - PAD.left - PAD.right;
const PLOT_H = H - PAD.top - PAD.bottom;

const SERIES_COLORS = {
  intensity: "#c07f2e",
  polarity: "#3f6f63",
  confidence: "#7a5aa6",
};

const wrapStyle = {
  border: "1px solid var(--border, #e5e5e5)",
  borderRadius: 8,
  padding: "10px 12px",
  background: "var(--surface, #fff)",
};
const legendStyle = { display: "flex", gap: 14, flexWrap: "wrap", marginTop: 6, fontSize: 12, color: "var(--ink-soft, #666)" };
const axisText = { fontSize: 10, fill: "var(--ink-faint, #8a9086)" };

function num(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function fmtNum(v) {
  return Number.isInteger(v) ? String(v) : v.toFixed(2);
}

function shortLabel(value) {
  if (value === null || value === undefined) return "";
  const s = String(value);
  if (/^\d{4}-\d{2}-\d{2}T/.test(s)) {
    const d = new Date(s);
    if (!Number.isNaN(d.getTime())) return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  }
  return s.length > 10 ? s.slice(5, 10) || s.slice(0, 10) : s;
}

function EmptyChart({ message }) {
  return <p className="hint">{message}</p>;
}

function Legend({ items }) {
  return (
    <div style={legendStyle}>
      {items.map((it) => (
        <span key={it.label} style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          <span style={{ width: 10, height: 10, borderRadius: 2, background: it.color, display: "inline-block" }} />
          {it.label}
        </span>
      ))}
    </div>
  );
}

// Generic multi-series line chart. points: [{ label, tooltip, values: { key: number|null } }]
function LineChart({ points, series, ariaLabel }) {
  const all = points.flatMap((p) => series.map((s) => p.values[s.key])).filter((v) => v !== null);
  const yMin = Math.min(0, ...all);
  const yMax = Math.max(1, ...all);
  const span = yMax - yMin || 1;
  const x = (i) => PAD.left + (points.length === 1 ? PLOT_W / 2 : (i / (points.length - 1)) * PLOT_W);
  const y = (v) => PAD.top + PLOT_H - ((v - yMin) / span) * PLOT_H;

  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => yMin + t * span);
  const labelEvery = Math.max(1, Math.ceil(points.length / 8));

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={ariaLabel} style={{ display: "block" }}>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={PAD.left} x2={W - PAD.right} y1={y(t)} y2={y(t)} stroke="var(--border, #e5e5e5)" strokeWidth="1" />
          <text x={PAD.left - 6} y={y(t) + 3} textAnchor="end" style={axisText}>{fmtNum(Number(t.toFixed(2)))}</text>
        </g>
      ))}
      {yMin < 0 ? (
        <line x1={PAD.left} x2={W - PAD.right} y1={y(0)} y2={y(0)} stroke="var(--ink-faint, #8a9086)" strokeDasharray="4 3" />
      ) : null}

      {series.map((s) => {
        const segs = [];
        let current = [];
        points.forEach((p, i) => {
          const v = p.values[s.key];
          if (v === null) {
            if (current.length) segs.push(current);
            current = [];
          } else current.push([x(i), y(v)]);
        });
        if (current.length) segs.push(current);
        return (
          <g key={s.key}>
            {segs.map((seg, i) => (
              <polyline
                key={i}
                points={seg.map((p) => p.join(",")).join(" ")}
                fill="none"
                stroke={s.color}
                strokeWidth="2"
                strokeLinejoin="round"
                strokeLinecap="round"
              />
            ))}
            {points.map((p, i) =>
              p.values[s.key] === null ? null : (
                <circle key={i} cx={x(i)} cy={y(p.values[s.key])} r="3.5" fill={s.color}>
                  <title>{`${p.tooltip} — ${s.label}: ${fmtNum(p.values[s.key])}`}</title>
                </circle>
              )
            )}
          </g>
        );
      })}

      {points.map((p, i) =>
        i % labelEvery === 0 ? (
          <text key={i} x={x(i)} y={H - PAD.bottom + 16} textAnchor="middle" style={axisText}>{p.label}</text>
        ) : null
      )}
    </svg>
  );
}

export function EmotionTrendChart({ rows }) {
  const points = useMemo(
    () =>
      (rows || []).map((r) => ({
        label: shortLabel(r.period),
        tooltip: `${r.period} (${r.samples ?? "?"} samples)`,
        values: { intensity: num(r.avg_intensity), polarity: num(r.avg_polarity) },
      })),
    [rows]
  );
  if (!points.length) return <EmptyChart message="No trend data to chart yet." />;
  const series = [
    { key: "intensity", label: "Avg intensity", color: SERIES_COLORS.intensity },
    { key: "polarity", label: "Avg polarity", color: SERIES_COLORS.polarity },
  ];
  return (
    <div style={wrapStyle}>
      <LineChart points={points} series={series} ariaLabel="Emotional trend over time" />
      <Legend items={series} />
    </div>
  );
}

export function EmotionFrequencyChart({ rows }) {
  const items = useMemo(
    () =>
      (rows || [])
        .map((r) => ({ emotion: r.emotion, count: num(r.count) || 0 }))
        .sort((a, b) => b.count - a.count),
    [rows]
  );
  if (!items.length) return <EmptyChart message="No emotion frequency data to chart yet." />;

  const max = Math.max(...items.map((i) => i.count), 1);
  const total = items.reduce((s, i) => s + i.count, 0) || 1;
  const rowH = 30;
  const labelW = 110;
  const chartH = items.length * rowH + 8;
  const barW = W - labelW - 60;

  return (
    <div style={wrapStyle}>
      <svg viewBox={`0 0 ${W} ${chartH}`} width="100%" role="img" aria-label="Emotion frequency" style={{ display: "block" }}>
        {items.map((it, i) => {
          const w = (it.count / max) * barW;
          const yPos = i * rowH + 4;
          return (
            <g key={it.emotion}>
              <text x={labelW - 8} y={yPos + 17} textAnchor="end" style={{ fontSize: 12, fill: "var(--ink, #222)" }}>
                {emotionEmoji(it.emotion)} {emotionLabel(it.emotion)}
              </text>
              <rect x={labelW} y={yPos + 3} width={Math.max(w, 2)} height={rowH - 10} rx="3" fill={emotionColor(it.emotion)}>
                <title>{`${emotionLabel(it.emotion)}: ${it.count} (${Math.round((it.count / total) * 100)}%)`}</title>
              </rect>
              <text x={labelW + Math.max(w, 2) + 6} y={yPos + 17} style={{ fontSize: 12, fill: "var(--ink-soft, #666)" }}>
                {it.count} · {Math.round((it.count / total) * 100)}%
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

const MAX_SCORE_POINTS = 30;

export function EmotionScoreChart({ items }) {
  const [show, setShow] = useState({ intensity: true, confidence: true, polarity: true });

  const points = useMemo(() => {
    const rows = (items || [])
      .map((r) => ({ ...r, _t: r.timestamp ? new Date(r.timestamp).getTime() : NaN }))
      .sort((a, b) => (Number.isNaN(a._t) || Number.isNaN(b._t) ? 0 : a._t - b._t))
      .slice(-MAX_SCORE_POINTS);
    return rows.map((r, i) => {
      const when = r.timestamp ? new Date(r.timestamp) : null;
      const stamp = when && !Number.isNaN(when.getTime()) ? when.toLocaleString() : `#${i + 1}`;
      return {
        label: when && !Number.isNaN(when.getTime()) ? shortLabel(r.timestamp) : `#${i + 1}`,
        tooltip: `${stamp} · ${emotionLabel(r.dominant_emotion)}`,
        values: {
          intensity: num(r.intensity),
          confidence: num(r.confidence),
          polarity: num(r.polarity_score),
        },
      };
    });
  }, [items]);

  if (!points.length) return <EmptyChart message="No emotion scores to chart yet." />;

  const allSeries = [
    { key: "intensity", label: "Intensity", color: SERIES_COLORS.intensity },
    { key: "confidence", label: "Confidence", color: SERIES_COLORS.confidence },
    { key: "polarity", label: "Polarity score", color: SERIES_COLORS.polarity },
  ];
  const active = allSeries.filter((s) => show[s.key]);

  return (
    <div style={wrapStyle}>
      <LineChart points={points} series={active} ariaLabel="Emotion scores per analysis" />
      <div style={legendStyle}>
        {allSeries.map((s) => (
          <label key={s.key} style={{ display: "inline-flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
            <input
              type="checkbox"
              checked={show[s.key]}
              onChange={() => setShow((prev) => ({ ...prev, [s.key]: !prev[s.key] }))}
            />
            <span style={{ width: 10, height: 10, borderRadius: 2, background: s.color, display: "inline-block" }} />
            {s.label}
          </label>
        ))}
        <span style={{ marginLeft: "auto" }}>Last {points.length} analyses</span>
      </div>
    </div>
  );
}
