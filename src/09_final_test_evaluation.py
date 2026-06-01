import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import csr_matrix, hstack
from sentence_transformers import SentenceTransformer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

    # ===== Meta features =====
    meta_features = ["title_length", "selftext_length", "hour", "weekday"]

    X_meta_train = train_df[meta_features].values.astype(float)
    X_meta_test = test_df[meta_features].values.astype(float)

    scaler = StandardScaler()
    X_meta_train = scaler.fit_transform(X_meta_train)
    X_meta_test = scaler.transform(X_meta_test)

    X_meta_train = csr_matrix(X_meta_train)
    X_meta_test = csr_matrix(X_meta_test)

    # ===== BERT embeddings =====
    print("Loading Sentence-BERT model...")
    bert_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    print("Encoding train texts...")
    X_bert_train = bert_model.encode(
        train_df["text"].tolist(),
        batch_size=32,
        show_progress_bar=True,
    )

    print("Encoding test texts...")
    X_bert_test = bert_model.encode(
        test_df["text"].tolist(),
        batch_size=32,
        show_progress_bar=True,
    )

    X_bert_train = csr_matrix(X_bert_train)
    X_bert_test = csr_matrix(X_bert_test)

    # ===== Final feature set: Meta + BERT =====
    X_train = hstack([X_meta_train, X_bert_train])
    X_test = hstack([X_meta_test, X_bert_test])

    # ===== Final model: XGBoost =====
    model = XGBClassifier(
        objective="multi:softmax",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=-1,
    )

    print("Training final XGBoost model...")
    model.fit(X_train, y_train)

    print("Evaluating on final test set...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    # ===== Save final results =====
    final_results = pd.DataFrame([
        {
            "model": "XGBoost",
            "features": "Meta + BERT",
            "test_size": 0.2,
            "test_samples": len(y_test),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        }
    ])

    final_results.to_csv(
        RESULTS_DIR / "final_test_results.csv",
        index=False,
    )

    print("\n===== Final test results =====")
    print(final_results)

    # ===== Classification report =====
    report = classification_report(
        y_test,
        y_pred,
        target_names=["Low", "Medium", "High"],
    )

    (RESULTS_DIR / "final_classification_report.txt").write_text(
        report,
        encoding="utf-8",
    )

    print("\n===== Classification report =====")
    print(report)

    # ===== Confusion matrix =====
    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=["Low", "Medium", "High"],
        columns=["Low", "Medium", "High"],
    )

    cm_df.to_csv(RESULTS_DIR / "final_confusion_matrix.csv")

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

    plt.title("Final Confusion Matrix - Meta + BERT + XGBoost")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "final_confusion_matrix_xgboost_meta_bert.png",
        dpi=300,
    )

    print("\nFiles were saved to:")
    print(RESULTS_DIR / "final_test_results.csv")
    print(RESULTS_DIR / "final_classification_report.txt")
    print(RESULTS_DIR / "final_confusion_matrix.csv")
    print(FIGURES_DIR / "final_confusion_matrix_xgboost_meta_bert.png")

    plt.close()


if __name__ == "__main__":
    main()