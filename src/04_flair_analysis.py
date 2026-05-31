import matplotlib.pyplot as plt
import seaborn as sns

from utils import ensure_dirs, load_data, FIGURES_DIR, RESULTS_DIR


FLAIR_CLEAN_MAP = {
    "💩Post / Meme 😂": "Post / Meme",
    "🕴️ Politics 🕴️": "Politics",
    "📰 News 📰": "News",
    "🤬 Rant 🤬": "Rant",
    "🟥 Bratislava ⬜": "Bratislava",
    "❔ General Discussion ❔": "General Discussion",
    "🏅 Sport 🏅": "Sport",
    "♘ Modrý koník ♘": "Modry konik",
}


def main() -> None:
    ensure_dirs()

    df = load_data()

    df["flair"] = df["flair"].fillna("No Flair").astype(str)
    df["flair"] = df["flair"].replace(FLAIR_CLEAN_MAP)

    top_flairs = df["flair"].value_counts().head(10).index
    df["flair_grouped"] = df["flair"].apply(
        lambda value: value if value in top_flairs else "Other"
    )

    results = (
        df.groupby("flair_grouped")["score"]
        .agg(["count", "mean", "median", "max"])
        .reset_index()
        .sort_values(by="median", ascending=False)
    )

    results.to_csv(RESULTS_DIR / "flair_analysis.csv", index=False)
    print(results)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=results, x="flair_grouped", y="median")
    plt.title("Medián popularity podľa flair kategórie")
    plt.xlabel("Flair")
    plt.ylabel("Medián skóre")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "flair_popularity.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
