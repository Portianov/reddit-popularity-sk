import praw
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent="script:reddit_popularity:v1.0 (by u/tvoj_username)"
)

subreddit = reddit.subreddit("Slovakia")

posts = []

for post in subreddit.hot(limit=100):
    posts.append({
        "title": post.title,
        "score": post.score,
        "num_comments": post.num_comments,
        "created_utc": post.created_utc,
        "upvote_ratio": post.upvote_ratio,
        "selftext": post.selftext,
        "flair": post.link_flair_text
    })

df = pd.DataFrame(posts)
df.to_csv("reddit_slovakia.csv", index=False, encoding='utf-8-sig')

print("CSV saved: reddit_slovakia.csv")
