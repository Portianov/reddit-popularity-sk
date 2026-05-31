# Reddit Popularity Prediction for Slovak Reddit Posts

This repository contains the code, dataset and experiment outputs used for a bachelor thesis focused on predicting the popularity of Reddit posts related to Slovakia.

The project studies whether the popularity of a Reddit post can be predicted from its text and metadata. Popularity is treated as a three-class classification problem:

- `0` — low popularity,
- `1` — medium popularity,
- `2` — high popularity.

The classes are created from the Reddit `score` column using the 25th and 75th percentiles.

## Main contribution

The repository supports the thesis contribution in three ways:

1. It contains a dataset of Reddit posts related to Slovakia and Slovak online communities.
2. It compares several feature representations: metadata, TF-IDF and multilingual Sentence-BERT embeddings.
3. It includes additional analysis of factors related to popularity, such as publication time, content type and flair category.

The project is focused on the Slovak and partially multilingual Reddit environment. This is relevant because the dataset contains posts from Slovak subreddits as well as international subreddits filtered by Slovakia-related keywords.

## Repository structure

```text
.
├── data/
│   └── reddit_slovakia_raw_big_20260220_101852.csv
├── figures/
│   └── generated figures used in the thesis
├── results/
│   └── CSV/TXT outputs of the experiments
├── src/
│   ├── 01_fetch_posts.py
│   ├── 02_time_analysis.py
│   ├── 03_post_type_analysis.py
│   ├── 04_flair_analysis.py
│   ├── 05_feature_importance.py
│   ├── 06_feature_comparison_logreg.py
│   ├── 07_compare_all_models.py
│   ├── 08_confusion_matrix.py
│   └── utils.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Dataset

The final raw dataset is stored in:

```text
data/reddit_slovakia_raw_big_20260220_101852.csv
```

It contains public Reddit post metadata with the following fields:

- `id`
- `subreddit`
- `title`
- `score`
- `num_comments`
- `created_utc`
- `upvote_ratio`
- `selftext`
- `flair`
- `url`

The target variable is not stored directly in the raw dataset. It is created in the scripts from the `score` column.

## Models and features

The experiments compare the following models:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- XGBoost

The following feature groups are evaluated:

- metadata only,
- TF-IDF only,
- BERT embeddings only,
- metadata + TF-IDF,
- metadata + BERT,
- TF-IDF + BERT,
- all features combined.

Text embeddings are generated using:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Reddit API credentials

The data collection script uses Reddit API credentials. Do not commit real credentials to GitHub.

Create a local `.env` file based on `.env.example`:

```bash
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USER_AGENT=reddit_popularity:v1.0 (by u/your_username)
```

## Running the scripts

Run scripts from the repository root.

Time analysis:

```bash
python src/02_time_analysis.py
```

Post type analysis:

```bash
python src/03_post_type_analysis.py
```

Flair analysis:

```bash
python src/04_flair_analysis.py
```

Feature importance:

```bash
python src/05_feature_importance.py
```

Comparison of feature combinations using Logistic Regression:

```bash
python src/06_feature_comparison_logreg.py
```

Comparison of all models and feature combinations:

```bash
python src/07_compare_all_models.py
```

Confusion matrix for the best configuration:

```bash
python src/08_confusion_matrix.py
```

## Notes

The `.env` file and IDE files are intentionally excluded from version control.  
The repository should not contain Reddit API secrets.
