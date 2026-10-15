import { useState } from "react";
import { useAuth } from "../auth/AuthContext.jsx";

const LINKS = [
  { to: "/", label: "Home" },
  { to: "/app", label: "Check in" },
  { to: "/dashboard", label: "Dashboard" },
  { to: "/about", label: "About" },
];

export default function SiteHeader({ route, theme, onToggleTheme }) {
  const [open, setOpen] = useState(false);
  const { session, isGuest, logout } = useAuth();

  return (
    <header className="masthead site-header">
      <a className="brand" href="#/" onClick={() => setOpen(false)}>
        <h1>MOOD MENTOR</h1>
        <p style={{ letterSpacing: "0.22em", fontWeight: 600, fontSize: 11, margin: "0 0 2px" }}>THE MIND COMPASS</p>
        <p>A quiet feedback loop for how you're actually doing.</p>
      </a>

      <button
        type="button"
        className="menu-toggle btn btn-ghost btn-sm"
        aria-expanded={open}
        aria-controls="site-nav"
        onClick={() => setOpen((v) => !v)}
      >
        {open ? "Close" : "Menu"}
      </button>

      <div id="site-nav" className={`site-nav${open ? " open" : ""}`}>
        <nav className="nav-tabs" aria-label="Primary">
          {LINKS.map((l) => (
            <a
              key={l.to}
              href={`#${l.to}`}
              className={route === l.to ? "active" : ""}
              aria-current={route === l.to ? "page" : undefined}
              onClick={() => setOpen(false)}
            >
              {l.label}
            </a>
          ))}
        </nav>

        <div className="header-actions">
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            onClick={onToggleTheme}
            aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
            title={theme === "dark" ? "Light mode" : "Dark mode"}
          >
            {theme === "dark" ? "☀️" : "🌙"}
          </button>
          {isGuest ? (
            <a className="btn btn-ghost btn-sm" href="#/login" onClick={() => setOpen(false)}>Sign in</a>
          ) : (
            <>
              <span className="pill" title={`Signed in (${session.mode})`}>👤 {session.username}</span>
              <button type="button" className="btn btn-ghost btn-sm" onClick={logout}>Sign out</button>
            </>
          )}
        </div>
      </div>
    </header>
  );
}
