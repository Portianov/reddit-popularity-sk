import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

#Nacitanie CSV bez stlpca score
df = pd.read_csv("reddit_slovakia_no_score.csv")

# Pridanie premennej is_weekend (1 = vikend, 0 = pracovny den)
df["is_weekend"] = df["weekday"].apply(lambda x: 1 if x in [5, 6] else 0)

# Vyber vstupnych priznakóv
features = ["title_length", "selftext_length", "hour", "weekday", "is_weekend"]
X = df[features]

# triedy popularity 0/1/2
y = df["popularity_multiclass"]

# Rozdelenie dat na trenovaciu a testovaciu mnozinu
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Trening modelu XGBoost
model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
model.fit(X_train, y_train)

# Vypocet dolezitosti priznakóv
importances = model.feature_importances_
importance_df = (
    pd.DataFrame({"Feature": features, "Importance": importances})
    .sort_values(by="Importance", ascending=False)
)

# Vizualizacia dolezitosti priznakóv
plt.figure(figsize=(8, 5))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Dolezitost")
plt.ylabel("Premenna")
plt.title("Dolezitost vstupnych premennych (bez score)")
plt.gca().invert_yaxis()  # nech najdolezitejsi je hore
plt.tight_layout()
plt.savefig("graf_feature_importance_no_score.png")
plt.show()
