const HELPLINE_URL = import.meta.env.VITE_HELPLINE_URL || "https://findahelpline.com";

export default function SiteFooter() {
  return (
    <footer className="site-footer">
      <p>
        <strong>MoodMentor is an educational wellness tool, not a medical device.</strong> It can't diagnose,
        treat or replace professional care. If you're in crisis, contact your local emergency number or{" "}
        <a href={HELPLINE_URL} target="_blank" rel="noreferrer">find a helpline</a>.
      </p>
      <nav aria-label="Footer" className="footer-links">
        <a href="#/about">About</a>
        <a href="#/privacy">Privacy</a>
        <a href="#/terms">Terms</a>
        <a href="#/dashboard">Delete my data</a>
      </nav>
      <p className="hint" style={{ margin: 0 }}>© {new Date().getFullYear()} MoodMentor</p>
    </footer>
  );
}
