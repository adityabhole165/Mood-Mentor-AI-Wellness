
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements_m4.txt
RUN python scripts/setup_nltk.py || true
EXPOSE 8501 8000
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
