export default function FeedbackLog({ entries }) {
  if (!entries?.length) return null;

  return (
    <div className="panel">
      <h2 style={{ fontSize: 16, marginBottom: 10 }}>Feedback logged this session</h2>
      <div className="history-list">
        {[...entries].reverse().map((e, i) => (
          <div className="history-row" key={i}>
            <span>
              {e.event} · {e.title}
            </span>
            <span className="hint">{new Date(e.timestamp).toLocaleTimeString()}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
