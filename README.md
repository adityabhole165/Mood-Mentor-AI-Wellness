# Emotion-Aware Recommendation System

An emotion-aware recommendation system built with Python. The project combines emotion detection, sentiment analysis, semantic matching, collaborative filtering, and hybrid recommendation techniques to generate personalized recommendations based on user interactions and emotional state.

## Project Structure

```text
.
├── src/
│   ├── build_ekman_dataset.py
│   ├── collaborative_filtering.py
│   ├── confidence.py
│   ├── data_loader.py
│   ├── emotional_state.py
│   ├── emotion_bert.py
│   ├── emotion_dataset.py
│   ├── emotion_distilbert.py
│   ├── evaluation.py
│   ├── fetch_isear.py
│   ├── hybrid_recommender.py
│   ├── ingestion.py
│   ├── interaction_service.py
│   ├── isear_validation.py
│   ├── milestone2_validation.py
│   ├── milestone3_validation.py
│   ├── personalized_recommender.py
│   ├── pipeline.py
│   ├── pipeline_v2.py
│   ├── pipeline_v3.py
│   ├── preprocessing.py
│   ├── ranking.py
│   ├── recommendation_data.py
│   ├── report.py
│   ├── semantic_matching.py
│   ├── sentiment.py
│   ├── train_models.py
│   └── __init__.py
│
├── README.md
├── .gitignore
└── requirements.txt
```

## Features

* Emotion detection from text
* Sentiment analysis
* Emotion-aware recommendation
* Personalized recommendations
* Collaborative filtering
* Hybrid recommendation
* Semantic matching
* Confidence scoring
* Data preprocessing and ingestion
* Model training and evaluation
* Validation scripts for different development milestones

## Requirements

Make sure Python is installed on your system.

Recommended Python version:

```text
Python 3.10+
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

If you do not have a `requirements.txt` file yet, you can create one after installing your project's dependencies:

```bash
pip freeze > requirements.txt
```

## Running the Project

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the appropriate Python module from the project root. For example:

```bash
python -m src.pipeline
```

Other available pipeline versions include:

```bash
python -m src.pipeline_v2
python -m src.pipeline_v3
```

Model training can be performed with:

```bash
python -m src.train_models
```

## Important

Do not commit generated Python cache files such as:

```text
__pycache__/
*.pyc
```

These files are automatically generated and should not be stored in Git.

Create a `.gitignore` file in the project root with:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class

# Virtual environments
venv/
.venv/
env/

# Environment variables
.env

# IDE files
.vscode/
.idea/

# OS files
.DS_Store

# Logs
*.log

# Jupyter
.ipynb_checkpoints/

# Build files
build/
dist/
*.egg-info/

# Model/data files
*.pkl
*.pickle
*.pt
*.pth
*.bin

# Temporary files
*.tmp
*.temp
```

## Push the Project to GitHub

### 1. Create a repository on GitHub

Go to GitHub and create a new repository.

For example:

```text
emotion-aware-recommendation-system
```

Do **not** initialize it with another README if you already have a local README.

### 2. Open the project folder in Terminal

Navigate to the folder containing:

```text
src/
README.md
.gitignore
```

For example:

```bash
cd path/to/your/project
```

### 3. Initialize Git

```bash
git init
```

### 4. Add the files

```bash
git add .
```

Check what will be committed:

```bash
git status
```

### 5. Create the first commit

```bash
git commit -m "Initial commit"
```

### 6. Connect the local project to GitHub

Replace `YOUR_USERNAME` and `YOUR_REPOSITORY` with your GitHub username and repository name:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

Verify the remote:

```bash
git remote -v
```

### 7. Push to GitHub

Set the main branch:

```bash
git branch -M main
```

Then push:

```bash
git push -u origin main
```

Your project should now appear on GitHub.

## Future Changes

After modifying your code, use:

```bash
git add .
git commit -m "Describe your changes"
git push
```

For example:

```bash
git add .
git commit -m "Improve emotion recommendation pipeline"
git push
```

## Typical Git Workflow

```bash
git status
git add .
git commit -m "Update project"
git push
```

## Troubleshooting

### Remote already exists

If you see:

```text
error: remote origin already exists
```

Check the existing remote:

```bash
git remote -v
```

If it is incorrect, change it:

```bash
git remote set-url origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

Then:

```bash
git push -u origin main
```

### GitHub asks for authentication

GitHub no longer accepts your normal account password for Git operations over HTTPS. Use GitHub authentication such as GitHub CLI or a Personal Access Token when prompted.

### Push rejected because the GitHub repository already has files

If the GitHub repository already contains a README or other initial commit, first run:

```bash
git pull origin main --allow-unrelated-histories
```

Resolve any conflicts if Git reports them, then:

```bash
git add .
git commit -m "Merge remote repository"
git push -u origin main
```

## License

Add the license appropriate for your project before publishing the repository.
