import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import csr_matrix, hstack
from sentence_transformers import SentenceTransformer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from xgboost import XGBClassifier

from utils import (
    ensure_dirs,
    load_data,
    add_basic_features,
    add_popularity_target,
    FIGURES_DIR,
    RESULTS_DIR,
)


def evaluate_final_model(model, X_train, X_test, y_train, y_test, model_name, feature_name):
    """Train model on training data and evaluate it once on final test data."""
    print(f"\nTraining final model: {model_name} + {feature_name}")
    model.fit(X_train, y_train)

    print("Evaluating on final test set...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Low", "Medium", "High"],
    )

    cm = confusion_matrix(y_test, y_pred)

    return {
        "model": model_name,
        "features": feature_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "classification_report": report,
        "confusion_matrix": cm,
    }


def save_confusion_matrix(cm, filename: str, title: str) -> None:
    """Save confusion matrix as PNG figure."""
    plt.figure(figsize=(7, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Low", "Medium", "High"],
        yticklabels=["Low", "Medium", "High"],
        annot_kws={"size": 12},
    )

    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()


def main() -> None:
    ensure_dirs()

    # ===== Load and prepare data =====
    df = load_data()
    df = add_basic_features(df)
    df = add_popularity_target(df)

    y = df["popularity_multiclass"].values

    # ===== Train / test split =====
    # Test set is used only once for final evaluation.
    train_df, test_df, y_train, y_test = train_test_split(
        df,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"Training samples: {len(train_df)}")
    print(f"Final test samples: {len(test_df)}")

    # ===== Meta features =====
    meta_features = ["title_length", "selftext_length", "hour", "weekday"]

    X_meta_train_dense = train_df[meta_features].values.astype(float)
    X_meta_test_dense = test_df[meta_features].values.astype(float)

    scaler = StandardScaler()
    X_meta_train_dense = scaler.fit_transform(X_meta_train_dense)
    X_meta_test_dense = scaler.transform(X_meta_test_dense)

    X_meta_train = csr_matrix(X_meta_train_dense)
    X_meta_test = csr_matrix(X_meta_test_dense)

    # ===== BERT embeddings =====
    print("Loading Sentence-BERT model...")
    bert_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    print("Encoding train texts...")
    X_bert_train_dense = bert_model.encode(
        train_df["text"].tolist(),
        batch_size=32,
        show_progress_bar=True,
    )

    print("Encoding test texts...")
    X_bert_test_dense = bert_model.encode(
        test_df["text"].tolist(),
        batch_size=32,
        show_progress_bar=True,
    )

    X_bert_train = csr_matrix(X_bert_train_dense)
    X_bert_test = csr_matrix(X_bert_test_dense)

    # ===== Feature sets for final evaluation =====
    X_meta_bert_train = hstack([X_meta_train, X_bert_train])
    X_meta_bert_test = hstack([X_meta_test, X_bert_test])

    # ===== Final models selected from validation results =====
    final_configs = [
        {
            "model_name": "Logistic Regression",
            "feature_name": "Meta + BERT",
            "model": make_pipeline(
                StandardScaler(with_mean=False),
                LogisticRegression(max_iter=3000, C=0.01),
            ),
            "X_train": X_meta_bert_train,
            "X_test": X_meta_bert_test,
        },
        {
            "model_name": "XGBoost",
            "feature_name": "Meta + BERT",
            "model": XGBClassifier(
                objective="multi:softmax",
                num_class=3,
                eval_metric="mlogloss",
                random_state=42,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                n_jobs=-1,
            ),
            "X_train": X_meta_bert_train,
            "X_test": X_meta_bert_test,
        },
        {
            "model_name": "Logistic Regression",
            "feature_name": "BERT",
            "model": make_pipeline(
                StandardScaler(with_mean=False),
                LogisticRegression(max_iter=3000, C=0.01),
            ),
            "X_train": X_bert_train,
            "X_test": X_bert_test,
        },
    ]

    final_results = []
    reports = []

    for config in final_configs:
        result = evaluate_final_model(
            model=config["model"],
            X_train=config["X_train"],
            X_test=config["X_test"],
            y_train=y_train,
            y_test=y_test,
            model_name=config["model_name"],
            feature_name=config["feature_name"],
        )

        final_results.append({
            "model": result["model"],
            "features": result["features"],
            "test_size": 0.2,
            "test_samples": len(y_test),
            "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"],
        })

        safe_model_name = result["model"].lower().replace(" ", "_")
        safe_feature_name = result["features"].lower().replace(" + ", "_").replace(" ", "_")

        reports.append(
            f"===== {result['model']} + {result['features']} =====\n"
            f"{result['classification_report']}\n"
        )

        cm_df = pd.DataFrame(
            result["confusion_matrix"],
            index=["Low", "Medium", "High"],
            columns=["Low", "Medium", "High"],
        )

        cm_df.to_csv(
            RESULTS_DIR / f"final_confusion_matrix_{safe_model_name}_{safe_feature_name}.csv"
        )

        save_confusion_matrix(
            cm=result["confusion_matrix"],
            filename=f"final_confusion_matrix_{safe_model_name}_{safe_feature_name}.png",
            title=f"Final Confusion Matrix - {result['model']} + {result['features']}",
        )

    # ===== Save final results =====
    final_results_df = pd.DataFrame(final_results).sort_values(
        by="macro_f1",
        ascending=False,
    )

    final_results_df.to_csv(
        RESULTS_DIR / "final_test_results.csv",
        index=False,
    )

    (RESULTS_DIR / "final_classification_reports.txt").write_text(
        "\n".join(reports),
        encoding="utf-8",
    )

    print("\n===== Final test results =====")
    print(final_results_df)

    print("\nFiles were saved to:")
    print(RESULTS_DIR / "final_test_results.csv")
    print(RESULTS_DIR / "final_classification_reports.txt")
    print(FIGURES_DIR)


if __name__ == "__main__":
    main()
