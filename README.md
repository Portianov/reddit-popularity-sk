# Predikcia popularity príspevkov na slovenskom Reddite

Tento repozitár obsahuje kód, dáta a výsledky experimentov k bakalárskej práci zameranej na predikciu popularity Reddit príspevkov súvisiacich so Slovenskom.

Cieľom projektu je zistiť, či je možné na základe textu príspevku a jeho metadát odhadnúť, do akej triedy popularity bude patriť.

Popularita je rozdelená do troch tried:

- `0` — nízka popularita,
- `1` — stredná popularita,
- `2` — vysoká popularita.

Triedy boli vytvorené zo stĺpca `score` pomocou 25. a 75. percentilu.

## Čo obsahuje projekt

Projekt obsahuje:

1. dataset Reddit príspevkov súvisiacich so Slovenskom,
2. skripty na spracovanie dát a vytvorenie príznakov,
3. porovnanie viacerých modelov strojového učenia,
4. analýzu faktorov, ktoré môžu súvisieť s popularitou príspevkov.

V práci boli použité hlavne tieto typy príznakov:

- meta-údaje,
- TF-IDF reprezentácia textu,
- Sentence-BERT embeddingy.

Okrem samotnej predikcie boli analyzované aj ďalšie faktory, napríklad čas publikovania, typ obsahu a flair kategórie.

## Štruktúra repozitára

```text
.
├── data/
│   └── reddit_slovakia_raw_big_20260220_101852.csv
├── figures/
│   └── grafy použité v práci
├── results/
│   └── výsledky experimentov
├── src/
│   ├── 01_fetch_posts.py
│   ├── 02_time_analysis.py
│   ├── 03_post_type_analysis.py
│   ├── 04_flair_analysis.py
│   ├── 05_feature_importance.py
│   ├── 06_feature_comparison_logreg.py
│   ├── 07_compare_all_models.py
│   ├── 08_confusion_matrix.py
│   └── utils.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Dataset

Hlavný dataset sa nachádza v súbore:

```text
data/reddit_slovakia_raw_big_20260220_101852.csv
```

Dataset obsahuje verejne dostupné údaje o Reddit príspevkoch, napríklad:

- `id`,
- `subreddit`,
- `title`,
- `score`,
- `num_comments`,
- `created_utc`,
- `upvote_ratio`,
- `selftext`,
- `flair`,
- `url`.

Cieľová premenná popularity sa vytvára v skriptoch zo stĺpca `score`.

## Použité modely

V experimentoch boli testované tieto modely:

- Logistic Regression,
- Decision Tree,
- Random Forest,
- Support Vector Machine,
- XGBoost.

Testované boli aj rôzne kombinácie príznakov:

- iba meta-údaje,
- iba TF-IDF,
- iba BERT embeddingy,
- meta-údaje + TF-IDF,
- meta-údaje + BERT,
- TF-IDF + BERT,
- všetky príznaky spolu.

Na vytvorenie embeddingov bol použitý model:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

## Inštalácia

Potrebné knižnice sa dajú nainštalovať pomocou:

```bash
pip install -r requirements.txt
```

Prípadne je možné najskôr vytvoriť virtuálne prostredie:

```bash
python -m venv .venv
```

Aktivácia vo Windows:

```bash
.venv\Scripts\activate
```

## Reddit API

Na zber nových dát je potrebné mať Reddit API údaje. Skutočné prihlasovacie údaje sa nemajú ukladať na GitHub.

V lokálnom projekte je potrebné vytvoriť súbor `.env` podľa súboru `.env.example`:

```env
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USER_AGENT=reddit_popularity:v1.0 (by u/your_username)
```

## Spustenie skriptov

Skripty sa spúšťajú z hlavného priečinka repozitára.

Analýza času publikovania:

```bash
python src/02_time_analysis.py
```

Analýza typu obsahu:

```bash
python src/03_post_type_analysis.py
```

Analýza flair kategórií:

```bash
python src/04_flair_analysis.py
```

Analýza dôležitosti príznakov:

```bash
python src/05_feature_importance.py
```

Porovnanie kombinácií príznakov:

```bash
python src/06_feature_comparison_logreg.py
```

Porovnanie všetkých modelov:

```bash
python src/07_compare_all_models.py
```

Confusion matrix pre najlepší model:

```bash
python src/08_confusion_matrix.py
```
