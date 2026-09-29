
from src.ingestion import ingest_raw_text, ingest_txt_file, ingest_csv_file
from src.preprocessing import preprocess_text
from src.sentiment import analyze_sentiment


def test_raw_txt_csv_ingestion(tmp_path):
    assert ingest_raw_text("hello").is_valid
    txt = tmp_path / "x.txt"; txt.write_text("hello\nworld", encoding="utf8")
    assert len(ingest_txt_file(str(txt))) == 2
    csv = tmp_path / "x.csv"; csv.write_text("text\nhello\nworld\n", encoding="utf8")
    assert len(ingest_csv_file(str(csv))) == 2


def test_preprocessing_edge_cases():
    out = preprocess_text("  Running,   runs!!! ")
    assert out.cleaned_text
    assert out.processed_text
    assert preprocess_text("!!!").is_empty_after_processing


def test_vader_dynamic_scores():
    pos = analyze_sentiment("I love this! Amazing!")
    neg = analyze_sentiment("I hate this. Terrible.")
    neu = analyze_sentiment("The meeting is at 3pm.")
    assert pos.compound > 0
    assert neg.compound < 0
    assert neu.label == "neutral"
    assert pos.compound != neg.compound
