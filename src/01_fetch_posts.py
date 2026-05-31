import os
from datetime import datetime
from time import sleep

import pandas as pd
import praw
import prawcore
from dotenv import load_dotenv


SLOVAK_SUBS = [
    "Slovakia",
    "Bratislava",
    "Kosice",
    "SlovakiaTravel",
]

BIG_SUBS = [
    "europe",
    "AskEurope",
    "travel",
    "expats",
]

FILTER_KEYWORDS = [
    "slovakia",
    "slovak",
    "slovensko",
    "bratislava",
    "kosice",
    "tatras",
    "high tatras",
    "low tatras",
]


def create_reddit_client() -> praw.Reddit:
    load_dotenv()

    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "reddit_popularity:v1.0")

    if not client_id or not client_secret:
        raise RuntimeError(
            "Missing Reddit API credentials. Create a .env file based on .env.example."
        )

    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )


def matches_filter(text: str) -> bool:
    text = (text or "").lower()
    return any(keyword in text for keyword in FILTER_KEYWORDS)


def add_post(posts: dict, post) -> None:
    posts[post.id] = {
        "id": post.id,
        "subreddit": str(post.subreddit),
        "title": post.title,
        "score": post.score,
        "num_comments": post.num_comments,
        "created_utc": post.created_utc,
        "upvote_ratio": getattr(post, "upvote_ratio", None),
        "selftext": getattr(post, "selftext", None),
        "flair": getattr(post, "link_flair_text", None),
        "url": post.url,
    }


def fetch_from_subreddit(reddit: praw.Reddit, posts: dict, sub_name: str, filtered: bool) -> None:
    print(f"\nCollecting from r/{sub_name} (filtered={filtered})")
    subreddit = reddit.subreddit(sub_name)

    fetch_configs = [
        ("top_all", subreddit.top, {"time_filter": "all", "limit": 2000}),
        ("top_year", subreddit.top, {"time_filter": "year", "limit": 2000}),
        ("top_month", subreddit.top, {"time_filter": "month", "limit": 2000}),
        ("new", subreddit.new, {"limit": 2000}),
        ("hot", subreddit.hot, {"limit": 1000}),
    ]

    before_count = len(posts)

    for label, fetch_func, kwargs in fetch_configs:
        print(f"  -> {label} ...", end="", flush=True)

        try:
            for post in fetch_func(**kwargs):
                if filtered:
                    text = f"{post.title or ''} {getattr(post, 'selftext', '') or ''}"
                    if not matches_filter(text):
                        continue
                add_post(posts, post)

            print(" ok")

        except (
            prawcore.exceptions.Forbidden,
            prawcore.exceptions.NotFound,
            prawcore.exceptions.Redirect,
        ) as exc:
            print(f" skipped ({exc.__class__.__name__})")

        except prawcore.PrawcoreException as exc:
            print(f" skipped (PrawcoreException: {exc})")

        sleep(1)

    print(f"  New posts from r/{sub_name}: {len(posts) - before_count}")


def main() -> None:
    reddit = create_reddit_client()
    posts = {}

    for sub in SLOVAK_SUBS:
        fetch_from_subreddit(reddit, posts, sub, filtered=False)

    for sub in BIG_SUBS:
        fetch_from_subreddit(reddit, posts, sub, filtered=True)

    df = pd.DataFrame(list(posts.values()))
    df = df.drop_duplicates(subset="id")

    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = f"{output_dir}/reddit_slovakia_raw_big_{timestamp}.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\nSaved {len(df)} unique posts to {output_path}")


if __name__ == "__main__":
    main()
