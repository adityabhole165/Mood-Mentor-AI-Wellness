"""Download the NLTK resources used by MoodMentor preprocessing."""
import nltk

for resource in ("stopwords", "wordnet", "omw-1.4", "punkt", "punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"):
    try:
        nltk.download(resource, quiet=False)
    except Exception as exc:
        print(f"Could not download {resource}: {exc}")
