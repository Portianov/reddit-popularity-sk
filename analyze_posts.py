import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# nacitavanie dat
df = pd.read_csv("reddit_slovakia.csv")

# prve 5 riadkov
print(df.head())

# vseobecne info
print(df.info())

# rozdelenie score
plt.figure(figsize=(10, 5))
sns.histplot(df['score'], bins=30, kde=True)
plt.title("Distribúcia popularity (score) príspevkov")
plt.xlabel("Počet upvote (score)")
plt.ylabel("Počet príspevkov")
plt.tight_layout()
plt.show()
