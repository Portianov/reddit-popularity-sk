from pathlib import Path

import pandas as pd


# Project root directory.
# This makes paths work both when scripts are started from the project root
# and when they are started directly from the src/ folder in PyCharm.
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"

DATA_PATH = DATA_DIR / "reddit_slovakia_raw_big_20260220_101852.csv"


def ensure_dirs() -> None:
    """Create output folders if they do not exist."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw Reddit dataset."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file was not found: {path}\n"
            "Check whether the CSV file is stored in the data/ folder."
        )

    return pd.read_csv(path)


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time, length and combined text features."""
    df = df.copy()

    df["created_utc"] = pd.to_datetime(
        df["created_utc"],
        unit="s",
        errors="coerce",
    )

    df["hour"] = df["created_utc"].dt.hour
    df["weekday"] = df["created_utc"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    df["title"] = df["title"].fillna("")
    df["selftext"] = df["selftext"].fillna("")

    df["title_length"] = df["title"].astype(str).str.len()
    df["selftext_length"] = df["selftext"].astype(str).str.len()

    df["text"] = (
        df["title"].astype(str)
        + " "
        + df["selftext"].astype(str)
    )

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