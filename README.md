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
2. skripty na zber a spracovanie dát,
3. porovnanie viacerých modelov strojového učenia,
4. porovnanie rôznych typov príznakov,
5. finálne vyhodnotenie najlepších konfigurácií na oddelenej testovacej množine,
6. analýzu faktorov, ktoré môžu súvisieť s popularitou príspevkov.

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
│   ├── 09_final_test_evaluation.py
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

Cieľová premenná popularity sa vytvára v skriptoch zo stĺpca `score`. Príspevky sú rozdelené do troch tried podľa 25. a 75. percentilu skóre.

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

## Metodika vyhodnotenia

Dataset bol rozdelený na dve časti:

- 80 % dát bolo použitých ako trénovacia a validačná časť,
- 20 % dát bolo ponechaných ako finálna testovacia množina.

Na trénovacej a validačnej časti bola použitá 5-násobná stratifikovaná krížová validácia. Tá slúžila na porovnanie kombinácií príznakov, modelov a vybraných hyperparametrov.

Finálna testovacia množina nebola použitá pri výbere modelu. Bola použitá až na záverečné vyhodnotenie najlepších konfigurácií.

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

Ak nastane problém s knižnicou `torch` vo Windows, je možné nainštalovať CPU verziu PyTorch samostatne:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
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

Porovnanie kombinácií príznakov pomocou Logistic Regression:

```bash
python src/06_feature_comparison_logreg.py
```

Porovnanie všetkých modelov a kombinácií príznakov:

```bash
python src/07_compare_all_models.py
```

Confusion matrix pre model Meta + BERT + XGBoost:

```bash
python src/08_confusion_matrix.py
```

Finálne vyhodnotenie najlepších konfigurácií na oddelenej testovacej množine:

```bash
python src/09_final_test_evaluation.py
```

## Výstupy

Výsledky experimentov sa ukladajú do priečinka:

```text
results/
```

Grafy sa ukladajú do priečinka:

```text
figures/
```

Medzi hlavné výstupy patria napríklad:

- `feature_comparison_results.csv`,
- `all_models_feature_comparison_results.csv`,
- `final_test_results.csv`,
- `final_classification_reports.txt`,
- grafy porovnania modelov,
- finálne confusion matrix grafy.

## Poznámka

Súbor `.env` nie je súčasťou repozitára, pretože môže obsahovať Reddit API kľúče. Na GitHube je uložený iba ukážkový súbor `.env.example`.
