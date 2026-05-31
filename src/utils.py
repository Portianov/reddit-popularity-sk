from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/reddit_slovakia_raw_big_20260220_101852.csv")
FIGURES_DIR = Path("figures")
RESULTS_DIR = Path("results")


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw Reddit dataset."""
    return pd.read_csv(path)


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time, length and combined text features."""
    df = df.copy()

    df["created_utc"] = pd.to_datetime(
        df["created_utc"],
        unit="s",
        errors="coerce"
    )

    df["hour"] = df["created_utc"].dt.hour
    df["weekday"] = df["created_utc"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    df["title"] = df["title"].fillna("")
    df["selftext"] = df["selftext"].fillna("")

    df["title_length"] = df["title"].astype(str).str.len()
    df["selftext_length"] = df["selftext"].astype(str).str.len()
    df["text"] = df["title"].astype(str) + " " + df["selftext"].astype(str)

    df = df.dropna(subset=["hour", "weekday", "score"])
    return df


def add_popularity_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create a 3-class popularity target based on score quantiles."""
    df = df.copy()

    q25 = df["score"].quantile(0.25)
    q75 = df["score"].quantile(0.75)

    def classify(score: float) -> int:
        if score <= q25:
            return 0  # low popularity
        if score <= q75:
            return 1  # medium popularity
        return 2      # high popularity

    df["popularity_multiclass"] = df["score"].apply(classify)
    return df
