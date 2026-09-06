"""
test_predict.py
Manual emotion-detection test harness for the Mood Mentor project.

Run from the project root (same folder as src/):
    python test_predict.py

What this does
--------------
Loads both fine-tuned models (BERT + DistilBERT) ONCE, then runs a fixed set
of sentences through both. The sentences are grouped into:

    1. SINGLE-EMOTION cases  -- one clear emotion each, one per Ekman category
    2. MULTI-EMOTION cases   -- deliberately mixed sentences (e.g. joy + fear)
    3. EDGE CASES            -- empty input, emojis, long text, ambiguous text

For each case it prints:
    - every emotion's raw score (sorted highest to lowest)
    - the primary (highest-scoring) emotion + confidence
    - which emotions crossed the 0.5 "triggered" threshold
    - a self-check against the emotion(s) you expected, so you can see at a
      glance whether the model actually agrees with you or not

Nothing here is hardcoded -- every score comes straight from the trained
model's predict() at inference time. This script does not modify or retrain
anything; it only loads existing weights from models/bert_emotion and
models/distilbert_emotion, so run `python -m src.train_models` first if
those folders don't exist yet.
"""
from __future__ import annotations

from src.emotion_bert import load_trained_model as load_bert, predict as predict_bert
from src.emotion_distilbert import load_trained_model as load_distil, predict as predict_distil
from src.emotion_dataset import EMOTIONS


# ---------------------------------------------------------------------------
# Test sentences
# ---------------------------------------------------------------------------
# "expected" is what a human would reasonably label the sentence as -- it's
# there so the script can flag agreement/disagreement, not because the model
# is required to match it exactly.

SINGLE_EMOTION_CASES = [
    ("joy",      "I just got the job offer, I'm absolutely thrilled!"),
    ("sadness",  "My dog passed away last week and I still can't stop crying."),
    ("anger",    "He lied to my face again and I am furious about it."),
    ("fear",     "I'm terrified about the results of my medical test tomorrow."),
    ("surprise", "I opened the door and my friends jumped out, I did not see that coming!"),
    ("disgust",  "The smell coming from that fridge made me want to throw up."),
]

MULTI_EMOTION_CASES = [
    (["joy", "fear"],
     "I am excited about the new opportunity but nervous about the outcome."),
    (["anger", "sadness"],
     "I'm so hurt and angry that he broke my trust like that."),
    (["surprise", "fear"],
     "Something just moved in the dark corner of the room and it scared me half to death."),
    (["joy", "surprise"],
     "I can't believe I actually won, this is the best surprise ever!"),
    (["disgust", "anger"],
     "It's disgusting how they treated the staff, it makes me so angry."),
]

EDGE_CASES = [
    ("empty string",        ""),
    ("whitespace only",     "    "),
    ("emojis / informal",   "lol i'm dead \U0001F602\U0001F602\U0001F602 that's actually hilarious"),
    ("very short",          "ok"),
    ("ambiguous / flat",    "The meeting is scheduled for 3pm on Thursday."),
    ("long rambling text",  "So today started off pretty normal, I made coffee, went to work, "
                             "nothing special, but then around lunch my manager pulled me aside "
                             "and told me the project I've been leading for six months got "
                             "cancelled out of nowhere, and honestly I don't even know how to "
                             "feel about it, part of me is relieved because it was exhausting "
                             "but another part of me is just really disappointed and a little "
                             "worried about what happens to my role now."),
]

THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run_one(label, text, model, tokenizer, predict_fn):
    try:
        result = predict_fn(text, model, tokenizer, threshold=THRESHOLD)
    except ValueError as e:
        print(f"    {label:<11s} -> rejected as invalid input ({e})")
        return None
    ranked = sorted(result.scores.items(), key=lambda kv: kv[1], reverse=True)
    scores_str = ", ".join(f"{e}={s:.2f}" for e, s in ranked)
    print(f"    {label:<11s} primary={result.primary_emotion:<9s} "
          f"conf={result.primary_confidence:.3f}  triggered={result.triggered_emotions}")
    print(f"    {'':<11s} all: {scores_str}")
    return result


def check_expected(expected, result, label):
    if result is None:
        return
    expected_set = set(expected) if isinstance(expected, list) else {expected}
    triggered_set = set(result.triggered_emotions)
    hit = expected_set & triggered_set
    missed = expected_set - triggered_set
    if missed:
        print(f"    {'':<11s} -> expected {sorted(expected_set)} | "
              f"MATCHED {sorted(hit) or '-'} | MISSED {sorted(missed)}")
    else:
        print(f"    {'':<11s} -> expected {sorted(expected_set)} | all triggered, matches")


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading trained models (once)...")
    bert_model, bert_tok = load_bert("models/bert_emotion")
    distil_model, distil_tok = load_distil("models/distilbert_emotion")
    print("Done.\n")

    tally = {"bert": {"single_hit": 0, "multi_full_hit": 0}, "distilbert": {"single_hit": 0, "multi_full_hit": 0}}

    # --- Single-emotion cases ---------------------------------------------
    section(f"SINGLE-EMOTION CASES ({len(SINGLE_EMOTION_CASES)} sentences, one Ekman emotion each)")
    for expected, text in SINGLE_EMOTION_CASES:
        print(f"\n  INPUT: {text!r}")
        b = run_one("BERT", text, bert_model, bert_tok, predict_bert)
        check_expected(expected, b, "BERT")
        if b and b.primary_emotion == expected:
            tally["bert"]["single_hit"] += 1
        d = run_one("DistilBERT", text, distil_model, distil_tok, predict_distil)
        check_expected(expected, d, "DistilBERT")
        if d and d.primary_emotion == expected:
            tally["distilbert"]["single_hit"] += 1

    # --- Multi-emotion cases ------------------------------------------------
    section(f"MULTI-EMOTION CASES ({len(MULTI_EMOTION_CASES)} sentences, 2 emotions expected each)")
    for expected, text in MULTI_EMOTION_CASES:
        print(f"\n  INPUT: {text!r}")
        b = run_one("BERT", text, bert_model, bert_tok, predict_bert)
        check_expected(expected, b, "BERT")
        if b and set(expected).issubset(set(b.triggered_emotions)):
            tally["bert"]["multi_full_hit"] += 1
        d = run_one("DistilBERT", text, distil_model, distil_tok, predict_distil)
        check_expected(expected, d, "DistilBERT")
        if d and set(expected).issubset(set(d.triggered_emotions)):
            tally["distilbert"]["multi_full_hit"] += 1

    # --- Edge cases -----------------------------------------------------------
    section(f"EDGE CASES ({len(EDGE_CASES)} inputs)")
    for label, text in EDGE_CASES:
        print(f"\n  [{label}] INPUT: {text!r}")
        run_one("BERT", text, bert_model, bert_tok, predict_bert)
        run_one("DistilBERT", text, distil_model, distil_tok, predict_distil)

    # --- Summary ----------------------------------------------------------
    section("SUMMARY")
    n_single = len(SINGLE_EMOTION_CASES)
    n_multi = len(MULTI_EMOTION_CASES)
    for name in ("bert", "distilbert"):
        s = tally[name]["single_hit"]
        m = tally[name]["multi_full_hit"]
        print(f"  {name:<11s} single-emotion primary match : {s}/{n_single}")
        print(f"  {'':<11s} multi-emotion  full match    : {m}/{n_multi}")
    print(f"\n  (Ekman categories checked: {EMOTIONS})")
    print("  'Full match' on multi-emotion cases means every expected emotion")
    print("  crossed the 0.5 threshold, not just the top-scoring one -- this is")
    print("  the strict check. Partial matches show up above as MATCHED/MISSED.")


if __name__ == "__main__":
    main()