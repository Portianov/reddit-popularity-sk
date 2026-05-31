# Dataset

This directory contains the final raw dataset used in the experiments:

- `reddit_slovakia_raw_big_20260220_101852.csv`

The dataset contains public Reddit post metadata related to Slovakia and Slovak online communities.  
Main fields: `id`, `subreddit`, `title`, `score`, `num_comments`, `created_utc`, `upvote_ratio`, `selftext`, `flair`, `url`.

The target variable is not stored in the raw file. It is created in the scripts from the `score` column using the 25th and 75th percentiles.
