const HELPLINE_URL = import.meta.env.VITE_HELPLINE_URL || "https://findahelpline.com";

// Shown automatically when a check-in reads as very intense / high severity,
// and available as a standing link in the footer.
export function shouldShowCrisis(state) {
  if (!state) return false;
  const severe = state.severity === "high" || state.severity === "critical";
  const heavy =
    (state.dominant_emotion === "sadness" || state.dominant_emotion === "fear") && Number(state.intensity) >= 0.8;
  return severe || heavy;
}

export default function CrisisBanner() {
  return (
    <div className="crisis-banner" role="region" aria-label="Support resources">
      <strong>You don't have to carry this alone.</strong>
      <p style={{ margin: "4px 0 0" }}>
        If you're in immediate danger or thinking about harming yourself, contact your local emergency number now.
        To find a free, confidential helpline in your country, visit{" "}
        <a href={HELPLINE_URL} target="_blank" rel="noreferrer">findahelpline.com</a>. MoodMentor is not a substitute
        for professional care.
      </p>
    </div>
  );
}
