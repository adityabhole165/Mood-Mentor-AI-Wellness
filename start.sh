#!/bin/bash

uvicorn src.api:app --host 0.0.0.0 --port 7860 &
streamlit run app.py --server.address=0.0.0.0 --server.port=8501 &

wait