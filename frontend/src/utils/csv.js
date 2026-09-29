// Minimal CSV helpers. Handles quoted fields with embedded commas/newlines
// (RFC 4180-ish) without pulling in a dependency, since MoodMentor's own
// sample CSVs (data/sample.csv, isear_subset.csv, ...) are simple.

export function parseCsv(rawText) {
  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;

  for (let i = 0; i < rawText.length; i++) {
    const char = rawText[i];
    const next = rawText[i + 1];

    if (inQuotes) {
      if (char === '"' && next === '"') {
        field += '"';
        i++;
      } else if (char === '"') {
        inQuotes = false;
      } else {
        field += char;
      }
      continue;
    }

    if (char === '"') {
      inQuotes = true;
    } else if (char === ",") {
      row.push(field);
      field = "";
    } else if (char === "\n" || char === "\r") {
      if (char === "\r" && next === "\n") i++;
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += char;
    }
  }
  if (field.length || row.length) {
    row.push(field);
    rows.push(row);
  }

  const nonEmpty = rows.filter((r) => r.some((cell) => cell.trim() !== ""));
  if (!nonEmpty.length) return { headers: [], records: [] };

  const headers = nonEmpty[0].map((h) => h.trim());
  const records = nonEmpty.slice(1).map((r) => {
    const record = {};
    headers.forEach((h, idx) => {
      record[h] = r[idx] ?? "";
    });
    return record;
  });
  return { headers, records };
}

// Picks the most likely free-text column: an exact "text" match first, else
// the first column whose average value length looks like prose rather than
// an id/number/short label.
export function guessTextColumn(headers, records) {
  if (!headers.length) return null;
  const lower = headers.map((h) => h.toLowerCase());
  const exact = lower.indexOf("text");
  if (exact !== -1) return headers[exact];
  const named = lower.findIndex((h) => ["input_text", "message", "entry", "content", "journal"].includes(h));
  if (named !== -1) return headers[named];

  let best = headers[0];
  let bestAvg = -1;
  for (const h of headers) {
    const avg =
      records.reduce((sum, r) => sum + String(r[h] || "").length, 0) / Math.max(1, records.length);
    if (avg > bestAvg) {
      bestAvg = avg;
      best = h;
    }
  }
  return best;
}

export function downloadCsv(filename, headers, rows) {
  const escape = (val) => {
    const str = String(val ?? "");
    return /[",\n]/.test(str) ? `"${str.replace(/"/g, '""')}"` : str;
  };
  const lines = [headers.join(","), ...rows.map((r) => headers.map((h) => escape(r[h])).join(","))];
  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
