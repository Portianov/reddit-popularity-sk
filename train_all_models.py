import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sentence_transformers import SentenceTransformer

#Nacitanie dat
df = pd.read_csv("reddit_slovakia_extended_multiclass.csv")

# Vytvorenie zakladnych priznakóv (meta-features)
df["title_length"] = df["title"].apply(lambda x: len(str(x)))
df["created_utc"] = pd.to_datetime(df["created_utc"], errors="coerce")
df["hour"] = df["created_utc"].dt.hour
df["day_of_week"] = df["created_utc"].dt.dayofweek
df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

# Cielova premenna (0 / 1 / 2)
y = df["popularity_multiclass"]

# Meta-priznaky (ciselne vlastnosti prispevku)
meta_features = df[["title_length", "hour", "day_of_week", "is_weekend"]]

# Normalizacia meta-priznakov
scaler = StandardScaler()
meta_scaled = scaler.fit_transform(meta_features)

# TF-IDF reprezentacia nadpisov
tfidf = TfidfVectorizer(max_features=300)
tfidf_features = tfidf.fit_transform(df["title"].astype(str)).toarray()

# BERT embeddingy (SentenceTransformer)
print("Nacitavam SentenceTransformer a generujem embeddingy...")
st_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = st_model.encode(df["title"].astype(str), show_progress_bar=True)

# Spojenie vsetkych priznakóv do jednej matice
X_full = np.concatenate([meta_scaled, tfidf_features, embeddings], axis=1)

# Balansovanie tried pomocou SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_full, y)

# Rozdelenie na trenovaciu a testovaciu mnozinu
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.2,
    random_state=42,
)

#Zoznam modelov na otestovanie
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
    ),
}

# Trening a vyhodnotenie modelov
for name, model in models.items():
    print(f"\n{name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Klasifikacna sprava:")
    print(classification_report(y_test, y_pred))

    # Konfuzna matica
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

#Krosvalidacia (5-fold cross-validation) na vyvazenych datach
print("\n===== Cross-validation (5-fold) =====")
for name, model in models.items():
    scores = cross_val_score(model, X_resampled, y_resampled, cv=5)
    print(f"{name}: {scores.mean():.3f} ± {scores.std():.3f}")
