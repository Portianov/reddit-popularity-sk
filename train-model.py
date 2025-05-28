import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# 1. Načítanie dát
df = pd.read_csv("reddit_slovakia.csv")

# 2. Základné spracovanie
df['created_dt'] = pd.to_datetime(df['created_utc'], unit='s')
df['hour'] = df['created_dt'].dt.hour
df['title_length'] = df['title'].apply(len)
df['is_popular'] = df['score'] > 50  # hranica popularity

# 3. Vyberieme iba niektoré stĺpce
features = df[['flair', 'hour', 'title_length']]
labels = df['is_popular'].astype(int)

# 4. Premeníme "flair" na čísla (OneHot)
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_flair = encoder.fit_transform(features[['flair']])

# Spojíme späť s ďalšími číselnými stĺpcami
X = np.concatenate([encoded_flair, features[['hour', 'title_length']].to_numpy()], axis=1)
y = labels.to_numpy()

# 5. Rozdelenie na trénovacie a testovacie dáta
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Tréning modelu
model = LogisticRegression()
model.fit(X_train, y_train)

# 7. Vyhodnotenie
y_pred = model.predict(X_test)

print("Presnosť:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))
