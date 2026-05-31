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
