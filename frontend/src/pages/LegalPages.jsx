const CONTACT = import.meta.env.VITE_CONTACT_EMAIL || "your-email@example.com";

function Page({ title, updated, children }) {
  return (
    <article className="panel prose">
      <h2 className="serif" style={{ fontSize: 28, marginBottom: 4 }}>{title}</h2>
      {updated ? <p className="hint" style={{ marginTop: 0 }}>Last updated: {updated}</p> : null}
      {children}
    </article>
  );
}

export function About() {
  return (
    <Page title="About MoodMentor">
      <p>
        MoodMentor is an educational project that turns short pieces of writing into an emotional read-out and a set of
        personalised wellness suggestions.
      </p>
      <h3>How the analysis works</h3>
      <ul>
        <li><strong>Preprocessing</strong> cleans and normalises your text.</li>
        <li><strong>VADER</strong> gives a baseline positive / negative / neutral sentiment.</li>
        <li><strong>BERT and DistilBERT</strong>, fine-tuned for multi-label emotion detection, score six emotions: joy, sadness, anger, fear, surprise and disgust. A single entry can carry several at once.</li>
        <li><strong>A hybrid recommender</strong> ranks wellness activities using rules, content similarity, sentence-embedding similarity, your preferences, collaborative filtering and your feedback history — and explains each pick.</li>
      </ul>
      <h3>Limitations</h3>
      <ul>
        <li>The models were trained on a small dataset and can be wrong, especially with sarcasm, slang, mixed languages and very short text.</li>
        <li>Emotion scores describe the <em>text</em>, not a clinical state. They are not a diagnosis.</li>
        <li>Word highlights show what the model reacted to, not what you meant.</li>
      </ul>
      <h3>Need support?</h3>
      <p>
        If you're struggling, please talk to someone you trust or a mental-health professional. In an emergency, contact
        your local emergency number.
      </p>
    </Page>
  );
}

export function Privacy() {
  return (
    <Page title="Privacy Policy" updated="Replace with your launch date">
      <p className="hint">
        Template for an educational project — review and adapt it (and the contact address) before you launch.
      </p>
      <h3>What we store</h3>
      <ul>
        <li><strong>Your entries' analysis</strong>: emotion scores, sentiment, recommendations shown, and the entry text used to generate them.</li>
        <li><strong>Your feedback</strong>: views, accepts, rejects, ratings and preference tags.</li>
        <li><strong>Your ID</strong>: a guest ID created in your browser, or the username you chose. Passwords for server accounts are stored only as salted hashes.</li>
        <li><strong>In your browser</strong>: your theme, your consent choice and your session, kept in local storage.</li>
      </ul>
      <h3>How it's used</h3>
      <p>
        Only to show your history and trends, and to personalise recommendations for you. We don't sell your data, show
        ads, or use your entries to train public models.
      </p>
      <h3>Third parties</h3>
      <p>
        Page fonts load from Google Fonts. The site and API are hosted on the providers named in the deployment
        documentation; they process requests on our behalf.
      </p>
      <h3>Your controls</h3>
      <ul>
        <li><strong>Export</strong> your data as CSV or JSON from the Dashboard.</li>
        <li><strong>Delete</strong> everything stored under your ID from the Dashboard (“Delete my data”).</li>
        <li><strong>Clear</strong> browser-side data by clearing this site's storage.</li>
      </ul>
      <h3>Contact</h3>
      <p>Questions or deletion requests: <a href={`mailto:${CONTACT}`}>{CONTACT}</a>.</p>
    </Page>
  );
}

export function Terms() {
  return (
    <Page title="Terms of Use" updated="Replace with your launch date">
      <p className="hint">
        Template for an educational project — review and adapt before you launch.
      </p>
      <h3>Not medical advice</h3>
      <p>
        MoodMentor is for self-reflection and education. It is not a medical device and does not provide diagnosis,
        treatment or professional advice. Don't use it in an emergency; contact your local emergency number or a
        helpline instead.
      </p>
      <h3>Acceptable use</h3>
      <p>
        Don't submit other people's private information, attempt to disrupt the service, or try to access data that
        isn't yours.
      </p>
      <h3>No warranty</h3>
      <p>
        The service is provided as-is. Predictions and recommendations may be inaccurate, and the service may be
        unavailable or reset at any time.
      </p>
      <h3>Changes</h3>
      <p>We may update these terms; continued use means you accept the updated version.</p>
    </Page>
  );
}
