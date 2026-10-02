import { useState } from "react";
import { useAuth } from "../auth/AuthContext.jsx";

export default function AuthPage({ mode, navigate }) {
  const { register, login, mode: authMode } = useAuth();
  const isSignup = mode === "signup";
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  async function submit(e) {
    e.preventDefault();
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      if (isSignup) await register(username, password);
      else await login(username, password);
      navigate("/app");
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="auth-wrap">
      <form className="panel auth-card" onSubmit={submit} noValidate>
        <h2 className="serif" style={{ fontSize: 24, marginBottom: 4 }}>{isSignup ? "Create your account" : "Welcome back"}</h2>
        <p className="hint" style={{ marginBottom: 16 }}>
          {isSignup
            ? "An account keeps your history under one name on this site."
            : "Sign in to see your history and trends."}
        </p>

        <div className="field">
          <label htmlFor="username">Username</label>
          <input id="username" autoComplete="username" value={username} onChange={(e) => setUsername(e.target.value)} required />
        </div>
        <div className="field">
          <label htmlFor="password">Password</label>
          <input
            id="password"
            type="password"
            autoComplete={isSignup ? "new-password" : "current-password"}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
          {isSignup ? <p className="hint" style={{ margin: "4px 0 0" }}>At least 8 characters.</p> : null}
        </div>

        {error ? <div className="error-banner" role="alert">{error}</div> : null}

        <button className="btn btn-primary" type="submit" disabled={busy || !username || !password}>
          {busy ? <span className="spinner" /> : null}
          {isSignup ? "Create account" : "Sign in"}
        </button>

        <p className="hint" style={{ marginTop: 14, marginBottom: 0 }}>
          {isSignup ? (
            <>Already have an account? <a href="#/login">Sign in</a></>
          ) : (
            <>New here? <a href="#/signup">Create an account</a></>
          )}
          {" · "}
          <a href="#/app">Continue as guest</a>
        </p>
        <p className="hint" style={{ marginTop: 8, marginBottom: 0 }}>
          {authMode === "guest" || authMode === "local"
            ? "Without the optional server auth, accounts are stored in this browser only."
            : null}
        </p>
      </form>
    </div>
  );
}
