export default function NotFound() {
  return (
    <div className="empty-state" style={{ marginTop: 20 }}>
      <h2 className="serif">404 — page not found</h2>
      <p>That page doesn't exist (or has moved).</p>
      <a className="btn btn-primary" href="#/" style={{ width: "auto", marginTop: 8 }}>Back to home</a>
    </div>
  );
}
