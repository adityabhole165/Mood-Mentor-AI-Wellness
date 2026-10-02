export default function ConsentCard({ checked, onChange }) {
  return (
    <div className="panel consent-card">
      <h2 style={{ fontSize: 16, marginBottom: 6 }}>Before you start</h2>
      <p style={{ margin: "0 0 10px", fontSize: 14 }}>
        MoodMentor stores your entries' analysis, your feedback and your emotion history on its server so it can show
        trends and learn your preferences. You can delete everything at any time from the Dashboard.
      </p>
      <label className="consent-label">
        <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
        <span>
          I've read the <a href="#/privacy">Privacy Policy</a> and <a href="#/terms">Terms</a>, and I understand
          MoodMentor is an educational tool, not medical advice.
        </span>
      </label>
    </div>
  );
}
