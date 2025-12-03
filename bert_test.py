import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("reddit_slovakia_extended_multiclass.csv")

# Text pre BERT (title + selftext)
df["text"] = df["title"].fillna("") + " " + df["selftext"].fillna("")
X_text = df["text"].tolist()

# Cielova premenna (triedy 0/1/2)
y = df["popularity_multiclass"].values

#Ciselne metadate prispevku
meta_features = ["title_length", "selftext_length", "hour", "weekday"]
X_meta = df[meta_features].values.astype(float)

#Načitanie modelu BERT (SentenceTransformer)
print("Nacitavam model BERT (SentenceTransformer all-MiniLM-L6-v2)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generujem BERT embeddingy pre text...")
X_bert = model.encode(X_text, batch_size=16, show_progress_bar=True)
X_bert = np.asarray(X_bert)

#Normalizacia metadat (scaling)
scaler = StandardScaler()
X_meta_scaled = scaler.fit_transform(X_meta)

#Spojenie BERT embeddingov a metadat do jedneho vektora
X_full = np.hstack([X_bert, X_meta_scaled])

print(f"Tvar BERT embeddingov: {X_bert.shape}")
print(f"Tvar metadat:          {X_meta_scaled.shape}")
print(f"Tvar spojenej matice:  {X_full.shape}")

#Rozdelenie na trenovaciu a testovaciu mnozinu
X_train, X_test, y_train, y_test = train_test_split(
    X_full,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,)

#Trening klasifikatora nad BERT + metadate
clf = LogisticRegression(max_iter=3000, n_jobs=-1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\n=== Vyhodnotenie modelu BERT + metadate (Logistic Regression) ===")
print(classification_report(y_test, y_pred))
