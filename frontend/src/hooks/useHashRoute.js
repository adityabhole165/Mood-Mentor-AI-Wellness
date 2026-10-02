import { useCallback, useEffect, useState } from "react";

// Tiny hash router: works on any static host (Vercel, Netlify, GitHub Pages)
// with no rewrite rules and no extra dependency.
function currentPath() {
  const raw = window.location.hash.replace(/^#/, "").split("?")[0];
  if (!raw) return "/";
  return raw.startsWith("/") ? raw : `/${raw}`;
}

export default function useHashRoute() {
  const [route, setRoute] = useState(currentPath);

  useEffect(() => {
    const onChange = () => {
      setRoute(currentPath());
      window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);

  const navigate = useCallback((to) => {
    window.location.hash = to;
  }, []);

  return [route, navigate];
}
