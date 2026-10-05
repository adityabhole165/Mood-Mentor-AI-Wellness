#!/usr/bin/env bash
# Run from the folder that contains Dockerfile (the "backend" folder), in Google Cloud Shell.
set -e
PROJECT_ID="${1:?Usage: ./deploy.sh YOUR_GCP_PROJECT_ID}"
REGION=asia-south1

gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

gcloud run deploy moodmentor-api \
  --source . \
  --region "$REGION" \
  --memory 4Gi --cpu 2 \
  --timeout 300 \
  --concurrency 4 \
  --min-instances 1 --max-instances 1 \
  --no-cpu-throttling --cpu-boost \
  --allow-unauthenticated

echo
echo "=== Your API URL ==="
gcloud run services describe moodmentor-api --region "$REGION" --format='value(status.url)'
