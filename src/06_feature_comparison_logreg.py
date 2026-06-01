import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix, hstack
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils import (
    ensure_dirs,
    load_data,
    add_basic_features,
    add_popularity_target,
    FIGURES_DIR,
    RESULTS_DIR,
)


def evaluate_model(model, X, y, name: str, cv) -> dict:
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )

    return {
        "features": name,
        "mean_f1": float(np.mean(scores)),
        "std_f1": float(np.std(scores)),
    }


def main() -> None:
    ensure_dirs()

    # ===== Load and prepare data =====
    df = load_data()
    df = add_basic_features(df)
    df = add_popularity_target(df)

    y = df["popularity_multiclass"].values

    # ===== Hold-out test split =====
    # The test set is not used in this script.
    # This script uses only the training/validation part for 5-fold CV.
    train_df, test_df, y_train, y_test = train_test_split(
        df,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"Training/validation samples: {len(train_df)}")
    print(f"Final test samples kept separate: {len(test_df)}")

    # ===== Features built only from training/validation data =====
    meta_features = ["title_length", "selftext_length", "hour", "weekday"]
    X_meta = csr_matrix(train_df[meta_features].values.astype(float))

    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(train_df["text"])

    print("Loading Sentence-BERT model...")
    bert_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    print("Encoding training/validation text...")
    X_bert_dense = bert_model.encode(
        train_df["text"].tolist(),
        batch_size=32,
        show_progress_bar=True,
    )
    X_bert = csr_matrix(X_bert_dense)

    feature_sets = {
        "Meta": X_meta,
        "TF-IDF": X_tfidf,
        "BERT": X_bert,
        "Meta + TF-IDF": hstack([X_meta, X_tfidf]),
        "Meta + BERT": hstack([X_meta, X_bert]),
        "TF-IDF + BERT": hstack([X_tfidf, X_bert]),
        "All combined": hstack([X_meta, X_tfidf, X_bert]),
    }

    model = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(max_iter=3000, C=0.01),
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = [
        evaluate_model(model, X, y_train, feature_name, cv)
        for feature_name, X in feature_sets.items()
    ]

    results_df = pd.DataFrame(results).sort_values(
        by="mean_f1",
        ascending=False,
    )

    results_df.to_csv(
        RESULTS_DIR / "feature_comparison_results.csv",
        index=False,
    )

    print(results_df)

    plot_df = results_df.sort_values(by="mean_f1", ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(plot_df["features"], plot_df["mean_f1"])
    plt.xlabel("Macro-F1")
    plt.ylabel("Kombinácia príznakov")
    plt.title("Porovnanie kombinácií príznakov")
    plt.grid(axis="x")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "feature_comparison_logreg.png",
        dpi=300,
    )

    plt.close()


if __name__ == "__main__":
    main()
