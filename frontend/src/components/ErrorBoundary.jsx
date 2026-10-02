import { Component } from "react";

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidUpdate(prevProps) {
    if (this.state.error && prevProps.resetKey !== this.props.resetKey) {
      this.setState({ error: null });
    }
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="panel" role="alert" style={{ marginTop: 20 }}>
        <h2 style={{ fontSize: 20, marginBottom: 6 }}>Something went wrong on this page</h2>
        <p className="hint" style={{ marginBottom: 14 }}>
          The rest of MoodMentor is fine. Try reloading, or head back to the start.
        </p>
        <div style={{ display: "flex", gap: 8 }}>
          <button className="btn btn-ghost" onClick={() => window.location.reload()}>Reload</button>
          <a className="btn btn-ghost" href="#/">Go home</a>
        </div>
      </div>
    );
  }
}
