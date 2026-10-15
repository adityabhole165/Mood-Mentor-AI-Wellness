// Client-side PDF export (no backend needed).
// Install once:  npm install jspdf jspdf-autotable
// The libraries are loaded on click, so they don't bloat the initial bundle.

function cell(value) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(3);
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

// sections: [{ title, columns: [{key,label}], rows: [{...}] }]
// summary : optional array of short strings printed under the title
export async function downloadPdf({ filename, title, summary = [], sections }) {
  const [{ jsPDF }, autoTableModule] = await Promise.all([
    import("jspdf"),
    import("jspdf-autotable"),
  ]);
  const autoTable = autoTableModule.default || autoTableModule.autoTable;

  const doc = new jsPDF({ orientation: "landscape", unit: "pt", format: "a4" });
  let y = 40;

  doc.setFontSize(18);
  doc.text(title, 40, y);
  y += 18;
  doc.setFontSize(9);
  doc.setTextColor(120);
  doc.text(`Generated ${new Date().toLocaleString()}`, 40, y);
  doc.setTextColor(0);
  y += 16;

  doc.setFontSize(11);
  summary.forEach((line) => {
    doc.text(line, 40, y);
    y += 15;
  });
  y += 6;

  sections.forEach((section) => {
    if (!section.rows?.length) return;
    doc.setFontSize(13);
    doc.text(section.title, 40, y);
    autoTable(doc, {
      startY: y + 8,
      head: [section.columns.map((c) => c.label)],
      body: section.rows.map((r) => section.columns.map((c) => cell(r[c.key]))),
      styles: { fontSize: 8, cellPadding: 4, overflow: "linebreak" },
      headStyles: { fillColor: [63, 111, 99] },
      margin: { left: 40, right: 40 },
    });
    y = doc.lastAutoTable.finalY + 24;
  });

  doc.save(filename);
}
