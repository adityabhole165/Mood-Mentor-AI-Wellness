FROM python:3.11-slim

RUN useradd -m -u 1000 user

WORKDIR /app

# Install CPU-only PyTorch
RUN pip install --no-cache-dir torch \
    --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
COPY requirements_m4.txt .
RUN pip install --no-cache-dir -r requirements_m4.txt

# Copy application
COPY --chown=user . /app

# NLTK setup
RUN python scripts/setup_nltk.py

USER user

# FastAPI + Streamlit
EXPOSE 7860 8501

# Start both through startup script
CMD ["bash", "start.sh"]