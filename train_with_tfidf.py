import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import classification_report

#Nacitanie dat
df = pd.read_csv("reddit_slovakia_extended_multiclass.csv")

#Odstranenie riadkov, kde chybaju dolezite hodnoty
df = df.dropna(subset=["title", "title_length", "weekday", "hour", "popularity_multiclass"])

# Vyber vstupnych priznakóv a cielovej premennej
X = df[["title", "title_length", "weekday", "hour"]]
y = df["popularity_multiclass"]

#Definicia textovych a numerickych priznakóv
text_features = "title"
numeric_features = ["title_length", "weekday", "hour"]


preprocessor = ColumnTransformer(
    transformers=[
        ("tfidf", TfidfVectorizer(max_features=500), text_features),
        ("num", StandardScaler(), numeric_features),
    ]
)

#Zoznam modelov na testovanie
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

#Rozdelenie dat na trenovaciu a testovaciu mnozinu
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

#Trening a hodnotenie modelov
for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print(f"\n{name}")
    print(classification_report(y_test, y_pred))

#5-fold krosvalidacia
print("\n===== Cross-validation (5-fold) =====")
for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    print(f"{name}: {scores.mean():.3f} ± {scores.std():.3f}")
