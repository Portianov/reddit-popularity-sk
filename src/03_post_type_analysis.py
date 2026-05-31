import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import ensure_dirs, load_data, FIGURES_DIR, RESULTS_DIR


def detect_post_type(url, selftext) -> str:
    url = str(url).lower()
    selftext = str(selftext).strip()

    if "i.redd.it" in url or url.endswith((".jpg", ".png", ".jpeg", ".webp")):
        return "Image"

    if ".gif" in url or "gifv" in url or "v.redd.it" in url:
        return "Video/GIF"

    if len(selftext) > 0 and "reddit.com" in url:
        return "Text"

    return "External Link"


def main() -> None:
    ensure_dirs()

    df = load_data()
    df["post_type"] = df.apply(
        lambda row: detect_post_type(row["url"], row["selftext"]),
        axis=1,
    )

    results = (
        df.groupby("post_type")["score"]
        .agg(["count", "mean", "median", "max"])
        .reset_index()
        .sort_values(by="mean", ascending=False)
    )

    results.to_csv(RESULTS_DIR / "post_type_analysis.csv", index=False)
    print(results)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=results, x="post_type", y="mean")
    plt.title("Priemerná popularita podľa typu obsahu")
    plt.xlabel("Typ obsahu")
    plt.ylabel("Priemerné skóre")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "post_type_popularity.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
