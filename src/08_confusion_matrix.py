import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import csr_matrix, hstack
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
    X_meta_dense = df[meta_features].values.astype(float)
    X_meta_dense = StandardScaler().fit_transform(X_meta_dense)
    X_meta = csr_matrix(X_meta_dense)

    print("Loading Sentence-BERT model...")
    bert_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    print("Encoding text...")
    X_bert_dense = bert_model.encode(
        df["text"].tolist(),
        batch_size=32,
        show_progress_bar=True,
    )
    X_bert = csr_matrix(X_bert_dense)

    X = hstack([X_meta, X_bert])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

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

    print("Training XGBoost...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred)
    (RESULTS_DIR / "classification_report_xgboost_meta_bert.txt").write_text(
        report,
        encoding="utf-8",
    )
    print(report)

    cm = confusion_matrix(y_test, y_pred)

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
    plt.title("Confusion Matrix - Meta + BERT + XGBoost")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix_xgboost_meta_bert.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
