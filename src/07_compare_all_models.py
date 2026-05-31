import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix, hstack
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from utils import (
    ensure_dirs,
    load_data,
    add_basic_features,
    add_popularity_target,
    FIGURES_DIR,
    RESULTS_DIR,
)


def main() -> None:
    ensure_dirs()

    df = load_data()
    df = add_basic_features(df)
    df = add_popularity_target(df)

    y = df["popularity_multiclass"].values

    meta_features = ["title_length", "selftext_length", "hour", "weekday"]
    X_meta = csr_matrix(df[meta_features].values.astype(float))

    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(df["text"])

    print("Loading Sentence-BERT model...")
    bert_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    print("Encoding text...")
    X_bert_dense = bert_model.encode(
        df["text"].tolist(),
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

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(max_iter=3000, C=0.01),
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
        "Random Forest": RandomForestClassifier(
            random_state=42,
            n_estimators=200,
            max_depth=15,
            n_jobs=-1,
        ),
        "SVM": make_pipeline(
            StandardScaler(with_mean=False),
            SVC(kernel="linear", C=1.0),
        ),
        "XGBoost": XGBClassifier(
            objective="multi:softmax",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            n_jobs=-1,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for feature_name, X in feature_sets.items():
        print(f"\nFeature set: {feature_name}")

        for model_name, model in models.items():
            print(f"Training {model_name}...")

            scores = cross_val_score(
                model,
                X,
                y,
                cv=cv,
                scoring="f1_macro",
                n_jobs=-1,
            )

            results.append({
                "features": feature_name,
                "model": model_name,
                "mean_f1": float(np.mean(scores)),
                "std_f1": float(np.std(scores)),
            })

    results_df = pd.DataFrame(results).sort_values(by="mean_f1", ascending=False)
    results_df.to_csv(RESULTS_DIR / "all_models_feature_comparison_results.csv", index=False)
    print(results_df)

    best_models = (
        results_df
        .sort_values("mean_f1", ascending=False)
        .drop_duplicates(subset="model", keep="first")
        .sort_values("mean_f1")
    )

    plt.figure(figsize=(9, 5))
    plt.plot(best_models["model"], best_models["mean_f1"], marker="o")
    plt.title("Porovnanie výkonnosti modelov")
    plt.xlabel("Model")
    plt.ylabel("Macro-F1")
    plt.ylim(0.45, 0.65)
    plt.grid(True)

    for i, value in enumerate(best_models["mean_f1"]):
        plt.text(i, value + 0.005, f"{value:.3f}", ha="center")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
