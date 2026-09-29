
# Implementation and demo guide

## M1 evidence
Run ingestion tests, preprocessing tests, VADER tests, and inspect `data/milestone1_report.csv`.

## M2 evidence
Run model evaluation/ISEAR validation with the trained model directories. The
project keeps model training separate from inference so the dashboard does not
silently retrain on every request.

## M3 evidence
1. Analyze a text.
2. Show emotional intensity/state.
3. Show history affinity.
4. Generate recommendations.
5. Submit feedback.
6. Show feedback-model status after enough labelled examples.
7. Open explanation evidence.
8. Run controlled Task 9 evaluation.
9. Verify `/analyze`, `/feedback`, `/stats` and SQLite persistence.

## M4 evidence
1. Open Dashboard/Reports tab.
2. Change daily/weekly/monthly trend granularity.
3. Filter by emotion/intensity and search records.
4. Inspect recommendation and feedback history.
5. Download CSV files and generate PDF.
6. Run stress test.
7. Demonstrate user-scoped deletion.
8. Run the complete pytest suite.
9. Build/install the package and run `moodmentor --help`.
10. Build the Docker image for deployment preparation.

## Important claims
Only report BERT/DistilBERT metric values from an actual evaluation run.
Only claim an advanced recommendation improvement if the controlled Task 9
metrics show it. Do not present synthetic demo interactions as real user data.


## Milestone 4 acceptance checklist

- Task 1: dashboard tab renders metrics and latest state.
- Task 2: daily/weekly/monthly trend aggregation is dynamic from stored timestamps.
- Task 3: recommendation and feedback records are loaded from SQLite and displayed.
- Task 4: search, emotion, intensity, recommendation-type and feedback-event filters work.
- Task 5: CSV downloads and PDF generation contain stored values.
- Task 6: `tests/test_milestone4_e2e.py` verifies the integrated component path.
- Task 7: `src/stress_test.py` reports average, p95 and max recommendation latency.
- Task 8: API validates user IDs/events; dashboard provides user-scoped deletion.
- Task 9: `pyproject.toml` exposes the `moodmentor` command.
- Task 10: README, API reference, Dockerfile and final test commands are included.
