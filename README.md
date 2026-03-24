# Predikcia popularity príspevkov na Reddite

Tento repozitár obsahuje materiály k mojej bakalárskej práci, ktorá sa zameriava na analýzu príspevkov zo sociálnej siete Reddit a predikciu ich popularity pomocou metód strojového učenia.

---

## Informácie o študentovi

- **Meno:** Yehor Portianov
- **Názov práce:** Predikcia popularity príspevkov na Reddite  
- **Vedúci práce:** Marek Šuppa
- **Kontakt:** portianov1@uniba.sk

---

## O čom je práca

V práci sa snažím odpovedať na otázku, prečo sú niektoré príspevky na Reddite úspešné a iné nie. Zameriavam sa na to, či sa dá popularita príspevku odhadnúť ešte pred jeho publikovaním na základe dostupných údajov.

Analyzujem príspevky zo subredditu r/Slovakia a skúmam rôzne faktory, ako napríklad text príspevku, čas publikovania alebo počet komentárov.

---

## Dáta

Používam vlastný dataset, ktorý obsahuje viac ako 3000 príspevkov.  
Každý príspevok obsahuje napríklad:

- názov
- text
- skóre
- počet komentárov  
- čas publikovania  
- kategóriu
- a t.d.

Tieto údaje tvoria základ pre ďalšiu analýzu a modelovanie.

---

## Ako postupujem

Najskôr som si dáta pripravil – očistil som ich a upravil do formy vhodnej pre analýzu.  
Následne som vytvoril nové vlastnosti (features), napríklad:

- dĺžku titulku  
- dĺžku textu  
- čas publikovania (hodina, deň v týždni)  

Textové údaje som previedol do číselnej podoby, aby ich bolo možné použiť v modeloch.

Na záver som testoval viaceré modely strojového učenia, napríklad:

- logistickú regresiu  
- random forest  
- SVM  
- XGBoost  

---

## Výsledky

Zistil som, že popularitu príspevkov je možné do určitej miery predpovedať.  
Najlepšie výsledky dosiahli modely Random Forest a XGBoost.

Zaujímavé je, že veľkú rolu nehrá len samotný obsah, ale aj napríklad čas publikovania alebo dĺžka príspevku.

---

## Na čo sa to dá využiť

Výsledky práce by sa dali využiť napríklad:

- pri tvorbe obsahu na sociálnych sieťach  
- v marketingu  
- na lepšie pochopenie správania používateľov  

---

## Použité technológie

Pri práci som použil najmä:

- Python  
- pandas, numpy  
- scikit-learn  
- XGBoost  
- Reddit API  

---

## Priebeh práce

Týždenný denník:

-24,03 Oprava Reddit API

-25.03 Dalšij zber dát z Redditu  
- ich spracovaniu a analýze
   
-30.03 testovaniu modelov BERT a GPT2  
- vyhodnotenie výsledkov  
