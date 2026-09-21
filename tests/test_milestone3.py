from src.emotional_state import analyze_emotional_state
from src.recommendation_data import DEFAULT_WELLNESS_CONTENT, UserProfile, Interaction
from src.semantic_matching import SemanticMatcher
from src.hybrid_recommender import HybridRecommendationEngine
from src.ranking import RecommendationRanker

def test_dynamic_intensity():
    low=analyze_emotional_state({"joy":.10,"sadness":.05,"anger":.02,"fear":.03,"surprise":.02,"disgust":.01},.02)
    high=analyze_emotional_state({"joy":.01,"sadness":.92,"anger":.10,"fear":.85,"surprise":.02,"disgust":.05},-.90)
    assert high.intensity>low.intensity and 0<=high.intensity<=1

def test_semantic_scores_exist():
    state=analyze_emotional_state({"joy":.01,"sadness":.80,"anger":.02,"fear":.70,"surprise":.01,"disgust":.01},-.6)
    scores=SemanticMatcher().match_emotional_state(state,"I feel anxious and overwhelmed",DEFAULT_WELLNESS_CONTENT)
    assert set(scores)=={c.content_id for c in DEFAULT_WELLNESS_CONTENT}

def test_dynamic_ranking_and_dedupe():
    state=analyze_emotional_state({"joy":.01,"sadness":.70,"anger":.02,"fear":.65,"surprise":.01,"disgust":.01},-.5)
    profile=UserProfile("u1",preferred_types=["breathing"],preferred_tags=["grounding"])
    semantic=SemanticMatcher().match_emotional_state(state,"I feel anxious",DEFAULT_WELLNESS_CONTENT)
    rows=HybridRecommendationEngine(DEFAULT_WELLNESS_CONTENT,[]).generate_candidates(state,profile,semantic)
    ranked=RecommendationRanker().rank(rows,state,5)
    assert ranked==sorted(ranked,key=lambda x:(-x.score,x.content_id))
    assert len({x.content_id for x in ranked})==len(ranked)
