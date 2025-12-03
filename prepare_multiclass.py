import pandas as pd
df = pd.read_csv("reddit_slovakia_extended_multiclass.csv")

if "score" not in df.columns:
    raise ValueError("V subore chyba stlpec 'score'. Skontroluj, ci je pritomny.")

#Vypocet percentilov (25% a 75%)
q25 = df["score"].quantile(0.25)
q75 = df["score"].quantile(0.75)

def classify(score):
    if score <= q25:
        return 0  # nizka popularita
    elif score <= q75:
        return 1  # stredna popularita
    else:
        return 2  # vysoka popularita

#Pridanie viac triednej premennej popularity
df["popularity_multiclass"] = df["score"].apply(classify)

df.to_csv("reddit_slovakia_extended_multiclass.csv", index=False)

print("Hotovo: bola pridana premenna 'popularity_multiclass'")
