import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { ApiError, authLogin, authRegister, setAuthToken } from "../api/client.js";

// Identity model
//  - guest   : random per-browser id (guest-xxxxxxxx). Nobody else's history is shared with you.
//  - server  : account on the MoodMentor backend (/auth routes present) — token-based.
//  - local   : demo account stored in THIS browser only (used when the backend has no /auth routes).
// The backend still trusts the user_id it is sent unless you add the optional auth extras.

const USERS_KEY = "moodmentor.users.v1";
const SESSION_KEY = "moodmentor.session.v1";
const GUEST_KEY = "moodmentor.guest.v1";
const USERNAME_RE = /^[A-Za-z0-9._-]{1,64}$/;
const PBKDF2_ITERATIONS = 150000;

const AuthContext = createContext(null);

function readJson(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

function writeJson(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // storage unavailable — session just won't persist
  }
}

function randomHex(bytes) {
  const arr = new Uint8Array(bytes);
  crypto.getRandomValues(arr);
  return Array.from(arr, (b) => b.toString(16).padStart(2, "0")).join("");
}

async function hashPassword(password, saltHex) {
  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey("raw", enc.encode(password), "PBKDF2", false, ["deriveBits"]);
  const salt = Uint8Array.from(saltHex.match(/.{2}/g).map((h) => parseInt(h, 16)));
  const bits = await crypto.subtle.deriveBits(
    { name: "PBKDF2", salt, iterations: PBKDF2_ITERATIONS, hash: "SHA-256" },
    key,
    256
  );
  return Array.from(new Uint8Array(bits), (b) => b.toString(16).padStart(2, "0")).join("");
}

function guestSession() {
  let id = readJson(GUEST_KEY, null);
  if (!id) {
    id = `guest-${randomHex(4)}`;
    writeJson(GUEST_KEY, id);
  }
  return { username: id, mode: "guest", token: null };
}

function initialSession() {
  const saved = readJson(SESSION_KEY, null);
  if (saved?.username && USERNAME_RE.test(saved.username)) return saved;
  return guestSession();
}

function validateCredentials(username, password) {
  if (!USERNAME_RE.test(username)) {
    throw new Error("Username must be 1–64 characters: letters, numbers, '.', '_' or '-'.");
  }
  if (password.length < 8) {
    throw new Error("Password must be at least 8 characters.");
  }
}

const isMissingRoute = (err) =>
  err instanceof ApiError && (err.status === 404 || err.status === 405 || err.status === 501);

export function AuthProvider({ children }) {
  const [session, setSession] = useState(initialSession);

  useEffect(() => {
    setAuthToken(session.token);
  }, [session.token]);

  const finish = useCallback((next) => {
    setSession(next);
    if (next.mode === "guest") {
      try {
        localStorage.removeItem(SESSION_KEY);
      } catch {
        // ignore
      }
    } else {
      writeJson(SESSION_KEY, next);
    }
  }, []);

  const register = useCallback(
    async (username, password) => {
      const name = username.trim();
      validateCredentials(name, password);
      try {
        const res = await authRegister({ username: name, password });
        finish({ username: res.username || name, mode: "server", token: res.token || null });
        return;
      } catch (err) {
        if (!isMissingRoute(err)) throw err;
      }
      const users = readJson(USERS_KEY, {});
      if (users[name]) throw new Error("That username is already taken on this device.");
      const salt = randomHex(16);
      users[name] = { salt, hash: await hashPassword(password, salt) };
      writeJson(USERS_KEY, users);
      finish({ username: name, mode: "local", token: null });
    },
    [finish]
  );

  const login = useCallback(
    async (username, password) => {
      const name = username.trim();
      validateCredentials(name, password);
      try {
        const res = await authLogin({ username: name, password });
        finish({ username: res.username || name, mode: "server", token: res.token || null });
        return;
      } catch (err) {
        if (err instanceof ApiError && err.status === 401) {
          throw new Error("Incorrect username or password.");
        }
        if (!isMissingRoute(err)) throw err;
      }
      const record = readJson(USERS_KEY, {})[name];
      if (!record || (await hashPassword(password, record.salt)) !== record.hash) {
        throw new Error("Incorrect username or password.");
      }
      finish({ username: name, mode: "local", token: null });
    },
    [finish]
  );

  const logout = useCallback(() => finish(guestSession()), [finish]);

  const value = useMemo(
    () => ({
      session,
      userId: session.username,
      isGuest: session.mode === "guest",
      mode: session.mode,
      register,
      login,
      logout,
    }),
    [session, register, login, logout]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}
