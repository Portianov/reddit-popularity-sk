import matplotlib.pyplot as plt
import seaborn as sns

from utils import ensure_dirs, load_data, add_basic_features, add_popularity_target, FIGURES_DIR


def main() -> None:
    ensure_dirs()

    df = load_data()
    df = add_basic_features(df)
    df = add_popularity_target(df)

    # Average class value by hour
    hour_popularity = df.groupby("hour")["popularity_multiclass"].mean()

    plt.figure(figsize=(8, 5))
    hour_popularity.plot(marker="o")
    plt.title("Priemerná popularita podľa hodiny publikovania")
    plt.xlabel("Hodina")
    plt.ylabel("Priemerná trieda popularity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "hour_average_popularity.png", dpi=300)
    plt.close()

    # Probability of high popularity by hour
    df["is_high"] = (df["popularity_multiclass"] == 2).astype(int)
    high_prob = df.groupby("hour")["is_high"].mean()

    plt.figure(figsize=(8, 5))
    high_prob.plot(marker="o")
    plt.title("Pravdepodobnosť vysokej popularity podľa hodiny")
    plt.xlabel("Hodina")
    plt.ylabel("Pravdepodobnosť")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "hour_high_popularity_probability.png", dpi=300)
    plt.close()

    # Heatmap: weekday vs hour
    pivot = df.pivot_table(
        values="is_high",
        index="weekday",
        columns="hour",
        aggfunc="mean",
    )

    plt.figure(figsize=(10, 5))
    sns.heatmap(pivot, cmap="coolwarm")
    plt.title("Pravdepodobnosť vysokej popularity podľa dňa a hodiny")
    plt.xlabel("Hodina")
    plt.ylabel("Deň v týždni")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "weekday_hour_heatmap.png", dpi=300)
    plt.close()

    print("Time-analysis figures saved to figures/.")


if __name__ == "__main__":
    main()
