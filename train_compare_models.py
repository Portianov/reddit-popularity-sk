import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Nacitanie rozsirenych dat s viac triednou (multiclass) popularitou
df = pd.read_csv("reddit_slovakia_extended_multiclass.csv")

# Vyber relevantnych vlastnosti
features = ["hour", "weekday", "selftext_length", "title_length"]
X = df[features]

# Cielova premenna (0 / 1 / 2)
y = df["popularity_class"]

# Rozdelenie dat na trenovaciu a testovaciu mnozinu
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalizacia ciselnych vlastnosti
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Zoznam modelov na otestovanie
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        multi_class="multinomial"
    ),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

# Trening a vyhodnotenie kazdeho modelu
for name, model in models.items():
    print(f"\n{name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
