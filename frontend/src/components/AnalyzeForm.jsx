import { useState } from "react";
import { parseCsv, guessTextColumn } from "../utils/csv.js";

const MODES = [
  { id: "chat", label: "💬 Chat text" },
  { id: "txt", label: "📄 TXT file" },
  { id: "csv", label: "📊 CSV file" },
];

const EXAMPLES = {
  "😰 Excited but nervous": "I am excited about the new opportunity but nervous about the outcome.",
  "😔 Low & overwhelmed":
    "I have been feeling really down and overwhelmed at work lately, nothing seems to go right.",
  "😠 Frustrated": "I'm furious that my manager took credit for my work again. I can't stop thinking about it.",
};

export default function AnalyzeForm({ onSubmitSingle, onSubmitBatch, isLoading, defaultUserId }) {
  const [mode, setMode] = useState("chat");
  const [text, setText] = useState("");
  const [userId, setUserId] = useState(defaultUserId);
  const [modelType, setModelType] = useState("BERT");
  const [topK, setTopK] = useState(5);
  const [fileName, setFileName] = useState("");
  const [csvPreview, setCsvPreview] = useState(null); // { headers, records, textColumn }
  const [maxRows, setMaxRows] = useState(20);
  const [fileError, setFileError] = useState(null);

  function handleModeChange(next) {
    setMode(next);
    setFileError(null);
    if (next !== "csv") setCsvPreview(null);
    if (next === "chat") {
      setFileName("");
      setText("");
    }
  }

  async function handleTxtFile(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setFileError(null);
    setFileName(file.name);
    try {
      setText(await file.text());
    } catch (err) {
      setFileError("Could not read that file as text.");
    }
  }

  async function handleCsvFile(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setFileError(null);
    setFileName(file.name);
    try {
      const raw = await file.text();
      const { headers, records } = parseCsv(raw);
      if (!records.length) {
        setFileError("That CSV has no rows.");
        setCsvPreview(null);
        return;
      }
      setCsvPreview({ headers, records, textColumn: guessTextColumn(headers, records) });
    } catch (err) {
      setFileError("Could not parse that CSV.");
      setCsvPreview(null);
    }
  }

  function handleSubmit(e) {
    e.preventDefault();
    if (isLoading) return;
    const uid = userId.trim() || "demo_user";
    const opts = { userId: uid, modelType, topK: Number(topK) };

    if (mode === "csv") {
      if (!csvPreview) return;
      const rows = csvPreview.records
        .map((r, i) => ({ id: i + 1, text: String(r[csvPreview.textColumn] || "").trim() }))
        .filter((r) => r.text)
        .slice(0, Number(maxRows));
      if (rows.length) onSubmitBatch({ rows, ...opts });
      return;
    }

    if (!text.trim()) return;
    onSubmitSingle({ text: text.trim(), ...opts });
  }

  const canSubmit = mode === "csv" ? Boolean(csvPreview?.records?.length) : Boolean(text.trim());

  return (
    <form className="panel panel-sticky" onSubmit={handleSubmit}>
      <h2 style={{ fontSize: 18, marginBottom: 4 }}>How are you doing?</h2>
      <p className="hint" style={{ marginBottom: 14 }}>
        Write a journal entry, or bring a file — MoodMentor reads the tone, not just the words.
      </p>

      <div className="nav-tabs" style={{ marginBottom: 14, width: "fit-content" }}>
        {MODES.map((m) => (
          <button type="button" key={m.id} className={mode === m.id ? "active" : ""} onClick={() => handleModeChange(m.id)}>
            {m.label}
          </button>
        ))}
      </div>

      {mode === "chat" && (
        <>
          <p className="hint" style={{ marginBottom: 8 }}>Try an example:</p>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 12 }}>
            {Object.entries(EXAMPLES).map(([label, sample]) => (
              <button
                type="button"
                key={label}
                className="btn btn-ghost btn-sm"
                onClick={() => setText(sample)}
              >
                {label}
              </button>
            ))}
          </div>
          <div className="field">
            <label htmlFor="text">Journal entry</label>
            <textarea
              id="text"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="I've been feeling..."
            />
          </div>
        </>
      )}

      {mode === "txt" && (
        <div className="field">
          <label htmlFor="txtFile">Upload a .txt file</label>
          <input id="txtFile" type="file" accept=".txt" onChange={handleTxtFile} />
          {fileName ? <p className="hint">Loaded: {fileName}</p> : null}
          {text ? (
            <textarea
              style={{ marginTop: 8 }}
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
          ) : null}
        </div>
      )}

      {mode === "csv" && (
        <div className="field">
          <label htmlFor="csvFile">Upload a CSV file</label>
          <input id="csvFile" type="file" accept=".csv" onChange={handleCsvFile} />
          {csvPreview ? (
            <>
              <p className="hint" style={{ marginTop: 6 }}>
                {csvPreview.records.length} row(s) found.
              </p>
              <label htmlFor="textCol" style={{ marginTop: 6 }}>
                Text column
              </label>
              <select
                id="textCol"
                value={csvPreview.textColumn || ""}
                onChange={(e) => setCsvPreview({ ...csvPreview, textColumn: e.target.value })}
              >
                {csvPreview.headers.map((h) => (
                  <option key={h} value={h}>
                    {h}
                  </option>
                ))}
              </select>
              <label htmlFor="maxRows" style={{ marginTop: 6 }}>
                Max rows to analyze
              </label>
              <input
                id="maxRows"
                type="number"
                min={1}
                max={200}
                value={maxRows}
                onChange={(e) => setMaxRows(e.target.value)}
              />
            </>
          ) : null}
        </div>
      )}

      {fileError ? <div className="error-banner">{fileError}</div> : null}

      <div className="field" style={{ marginTop: 4 }}>
        <label htmlFor="userId">User ID</label>
        <input id="userId" value={userId} onChange={(e) => setUserId(e.target.value)} />
      </div>

      <div className="field-row">
        <div className="field">
          <label htmlFor="modelType">Emotion model</label>
          <select id="modelType" value={modelType} onChange={(e) => setModelType(e.target.value)}>
            <option value="BERT">BERT</option>
            <option value="DistilBERT">DistilBERT</option>
          </select>
        </div>
        <div className="field">
          <label htmlFor="topK">Recommendations</label>
          <input
            id="topK"
            type="number"
            min={1}
            max={20}
            value={topK}
            onChange={(e) => setTopK(e.target.value)}
          />
        </div>
      </div>

      {mode === "chat" ? (
        <p className="hint" style={{ marginBottom: 10 }}>
          Runs BERT and DistilBERT side by side, then recommends from the selected model above.
        </p>
      ) : null}

      <button className="btn btn-primary" type="submit" disabled={isLoading || !canSubmit}>
        {isLoading ? <span className="spinner" /> : null}
        {isLoading
          ? "Reading between the lines…"
          : mode === "csv"
          ? "Analyze rows"
          : "Analyze & recommend"}
      </button>
    </form>
  );
}
