import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

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

    features = [
        "title_length",
        "selftext_length",
        "hour",
        "weekday",
        "is_weekend",
        "num_comments",
        "upvote_ratio",
    ]

    X = df[features].fillna(0)
    y = df["popularity_multiclass"]

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

    model.fit(X_train, y_train)

    importance_df = (
        pd.DataFrame({
            "feature": features,
            "importance": model.feature_importances_,
        })
        .sort_values(by="importance", ascending=False)
    )

    importance_df.to_csv(RESULTS_DIR / "feature_importance_meta.csv", index=False)
    print(importance_df)

    plt.figure(figsize=(8, 5))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.xlabel("Dôležitosť")
    plt.ylabel("Premenná")
    plt.title("Dôležitosť vybraných meta-príznakov")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance_meta.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
