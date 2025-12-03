import pandas as pd
from datetime import datetime

# Nacitanie povodnych dat
df = pd.read_csv("reddit_slovakia.csv", encoding="utf-8")

# Pridanie zakladnych vlastnosti prispevkov
df["title_length"] = df["title"].apply(lambda x: len(str(x)))
df["selftext_length"] = df["selftext"].apply(lambda x: len(str(x)) if pd.notnull(x) else 0)

# Konverzia casu z UNIX na datetime
df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s")

# Hodina publikacie
df["hour"] = df["created_utc"].dt.hour

# Den v tyzdni (0 = pondelok, 6 = nedela)
df["weekday"] = df["created_utc"].dt.weekday

# Vypocet percentilov pre skore
q25 = df["score"].quantile(0.25)
q75 = df["score"].quantile(0.75)

# Funkcia na rozdelenie popularity do 3 tried
def classify(score):
    if score <= q25:
        return 0  # nizka popularita
    elif score <= q75:
        return 1  # stredna popularita
    else:
        return 2  # vysoka popularita

# Pridanie cielovej premennej
df["popularity"] = df["score"].apply(classify)

# Ulozenie rozsirenych dat
df.to_csv("reddit_slovakia_extended.csv", index=False, encoding="utf-8")

# Kontrolny vypis
print(df[["score", "popularity", "title_length", "selftext_length", "hour", "weekday"]].head())
