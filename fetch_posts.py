import time
from typing import Dict, Any, List, Tuple

import praw
import prawcore
import pandas as pd

# Nastavenie Reddit API
reddit = praw.Reddit(
    client_id="k9EcZYyCzvVI7LphStYKxg",
    client_secret="Ys9e2CafoVH9KHvYeW3kTKST7eeOwQ",
    user_agent="reddit_popularity:v2.0 (by u/tvoj_username)",
)

# Zoznam subredditov
SUBREDDITS: List[str] = [
    # Slovenske subreddity
    "Slovakia",
    "Bratislava",
    "Kosice",
    "SlovakiaTravel",
    # Vacsie vseobecne subreddity (filtrovaie podla klucovych slov)
    "europe",
    "AskEurope",
    "travel",
    "VisitingEurope",
    "expats",
]

# Klucove slova pre Slovensko
FILTER_KEYWORDS: List[str] = [
    "slovakia",
    "slovak",
    "slovensko",
    "bratislava",
    "kosice",
    "tatras",
    "high tatras",
]


def matches_filter(post) -> bool:
    """Overi, ci prispevok obsahuje text o Slovensku."""
    title = post.title or ""
    body = post.selftext or ""
    text = (title + " " + body).lower()
    return any(keyword in text for keyword in FILTER_KEYWORDS)


# Slovnik pre ulozenie prispevkov (id -> data)
posts_dict: Dict[str, Dict[str, Any]] = {}


def add_post(post) -> None:
    """Prida prispevok do slovnika (deduplikacia podla id)."""
    posts_dict[post.id] = {
        "id": post.id,
        "subreddit": str(post.subreddit),
        "title": post.title,
        "score": post.score,
        "num_comments": post.num_comments,
        "created_utc": post.created_utc,
        "upvote_ratio": post.upvote_ratio,
        "selftext": post.selftext,
        "flair": post.link_flair_text,
        "url": post.url,
    }


def main() -> None:
    # Hlavny zber dat zo vsetkych subredditov
    for sub in SUBREDDITS:
        print(f"\n=== Zbieram data z r/{sub} ===")
        subreddit = reddit.subreddit(sub)

        before_count = len(posts_dict)

        try:
            # Rozne rezimy vyberu prispevkov
            fetch_configs: List[Tuple[str, Any, Dict[str, Any]]] = [
                ("top_all", subreddit.top, {"time_filter": "all", "limit": 5000}),
                ("top_year", subreddit.top, {"time_filter": "year", "limit": 5000}),
                ("new", subreddit.new, {"limit": 5000}),
            ]

            for label, func, kwargs in fetch_configs:
                print(f"  -> {label} ...", end="", flush=True)
                try:
                    for post in func(**kwargs):
                        # Filtrovanie podla klucovych slov
                        if matches_filter(post):
                            add_post(post)
                    print(" ok")
                except (
                    prawcore.exceptions.Forbidden,
                    prawcore.exceptions.NotFound,
                    prawcore.exceptions.Redirect,
                ) as e:
                    print(f" preskocene ({e.__class__.__name__})")
                    continue

            # Kratka pauza, aby sme nepretazili API
            time.sleep(1)

        except prawcore.PrawcoreException as e:
            print(f" Upozornenie: chyba pri praci s r/{sub}: {e}")
            continue

        after_count = len(posts_dict)
        print(f"  Novych prispevkov z r/{sub}: {after_count - before_count}")

    # Prevod do DataFrame
    df = pd.DataFrame(list(posts_dict.values()))

    print("\nSpolu prispevkov pred finalnym filtrom:", len(df))

    # Dodatocna kontrola filtra (pre istotu)
    mask = (
        df["title"].fillna("").str.lower().str.contains("slov|bratislav|kosice|tatras")
        | df["selftext"]
        .fillna("")
        .str.lower()
        .str.contains("slov|bratislav|kosice|tatras")
    )
    df = df[mask]

    print("Spolu prispevkov po finalnom filtri:", len(df))

    # Ulozenie vysledneho CSV
    df.to_csv("reddit_slovakia.csv", index=False, encoding="utf-8-sig")

    print("\nHotovo.")
    print("Spolu prispevkov ulozenych v subore:", len(df))


if __name__ == "__main__":
    main()
