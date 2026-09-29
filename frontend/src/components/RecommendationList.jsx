import RecommendationCard from "./RecommendationCard.jsx";

export default function RecommendationList({ recommendations, userId, onFeedback }) {
  if (!recommendations?.length) {
    return <p className="hint">No recommendations cleared the relevance threshold this time.</p>;
  }

  return (
    <div className="feed">
      {recommendations.map((rec) => (
        <RecommendationCard key={rec.content_id} rec={rec} userId={userId} onFeedback={onFeedback} />
      ))}
    </div>
  );
}
