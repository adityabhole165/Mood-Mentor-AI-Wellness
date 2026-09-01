
import streamlit as st
import pandas as pd
import tempfile
import os

from src.pipeline import run_pipeline, run_pipeline_multi


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Mood Mentor",
    page_icon="🧠",
    layout="wide"
)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("🧠 Mood Mentor")
st.subheader("AI Emotion Understanding & Personalized Wellness")

st.write(
    "Analyze text using preprocessing and VADER baseline sentiment analysis."
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("Mood Mentor")

mode = st.sidebar.radio(
    "Choose input method",
    [
        "💬 Chat Text",
        "📄 TXT File",
        "📊 CSV File"
    ]
)


# ---------------------------------------------------------
# CHAT TEXT
# ---------------------------------------------------------

if mode == "💬 Chat Text":

    st.header("💬 Analyze Text")

    text = st.text_area(
        "Enter your message or journal entry:",
        height=200,
        placeholder="Example: I have been feeling really anxious about work today..."
    )

    if st.button("🔍 Analyze Mood", type="primary"):

        if not text.strip():
            st.warning("Please enter some text first.")

        else:

            with st.spinner("Analyzing your mood..."):

                try:

                    df, summary = run_pipeline(
                        "raw_text",
                        text
                    )

                    st.success("Analysis completed!")

                    # -------------------------------
                    # SUMMARY
                    # -------------------------------

                    st.header("📈 Sentiment Summary")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            "Records",
                            summary.get("sample_size", len(df))
                        )

                    with col2:
                        st.metric(
                            "Average Compound",
                            round(
                                summary.get("avg_compound_score", 0),
                                3
                            )
                        )

                    with col3:
                        st.metric(
                            "Positive",
                            summary.get("positive_count", 0)
                        )

                    with col4:
                        st.metric(
                            "Negative",
                            summary.get("negative_count", 0)
                        )

                    # -------------------------------
                    # RESULTS
                    # -------------------------------

                    st.header("🧠 Analysis Result")

                    st.dataframe(
                        df,
                        use_container_width=True
                    )

                except Exception as e:

                    st.error(
                        f"Something went wrong during analysis: {e}"
                    )


# ---------------------------------------------------------
# TXT FILE
# ---------------------------------------------------------

elif mode == "📄 TXT File":

    st.header("📄 Upload Journal / Text File")

    uploaded_file = st.file_uploader(
        "Upload a .txt file",
        type=["txt"]
    )

    if uploaded_file is not None:

        st.info(
            f"Uploaded: {uploaded_file.name}"
        )

        if st.button("🔍 Analyze File", type="primary"):

            try:

                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".txt"
                ) as tmp:

                    tmp.write(uploaded_file.getvalue())
                    temp_path = tmp.name

                with st.spinner("Analyzing your file..."):

                    df, summary = run_pipeline(
                        "txt_file",
                        temp_path
                    )

                # Remove temporary file
                os.unlink(temp_path)

                st.success("Analysis completed!")

                st.header("📈 Summary")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Records",
                        summary.get("sample_size", len(df))
                    )

                with col2:
                    st.metric(
                        "Average Compound",
                        round(
                            summary.get("avg_compound_score", 0),
                            3
                        )
                    )

                with col3:
                    st.metric(
                        "Positive",
                        summary.get("positive_count", 0)
                    )

                with col4:
                    st.metric(
                        "Negative",
                        summary.get("negative_count", 0)
                    )

                st.header("📋 Detailed Results")

                st.dataframe(
                    df,
                    use_container_width=True
                )

            except Exception as e:

                st.error(
                    f"Unable to process the file: {e}"
                )


# ---------------------------------------------------------
# CSV FILE
# ---------------------------------------------------------

elif mode == "📊 CSV File":

    st.header("📊 Upload CSV Mood Data")

    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            preview_df = pd.read_csv(uploaded_file)

            st.write("### File Preview")

            st.dataframe(
                preview_df.head(),
                use_container_width=True
            )

            st.write(
                f"Rows: {len(preview_df)}"
            )

        except Exception as e:

            st.error(
                f"Could not read CSV: {e}"
            )

        if st.button("🔍 Analyze CSV", type="primary"):

            try:

                # Reset file position
                uploaded_file.seek(0)

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".csv"
                ) as tmp:

                    tmp.write(uploaded_file.getvalue())
                    temp_path = tmp.name

                with st.spinner("Analyzing CSV..."):

                    df, summary = run_pipeline(
                        "csv_file",
                        temp_path
                    )

                os.unlink(temp_path)

                st.success("CSV analysis completed!")

                st.header("📈 Summary")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Records",
                        summary.get("sample_size", len(df))
                    )

                with col2:
                    st.metric(
                        "Average Compound",
                        round(
                            summary.get("avg_compound_score", 0),
                            3
                        )
                    )

                with col3:
                    st.metric(
                        "Positive",
                        summary.get("positive_count", 0)
                    )

                with col4:
                    st.metric(
                        "Negative",
                        summary.get("negative_count", 0)
                    )

                st.header("📋 Detailed Results")

                st.dataframe(
                    df,
                    use_container_width=True
                )

            except Exception as e:

                st.error(
                    f"Unable to analyze CSV: {e}"
                )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "Mood Mentor — Milestone 1 | Text Ingestion + Preprocessing + VADER Sentiment"
)

