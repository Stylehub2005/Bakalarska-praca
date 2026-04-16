# 📊 Segmify – systém segmentácie zákazníkov

Webová aplikácia vytvorená v rámci bakalárskej práce, zameraná na analýzu transakčných dát, RFM segmentáciu, clustering zákazníkov, monitoring trendov a tvorbu marketingových odporúčaní.

🌐 **Online aplikácia:**  
https://bakalarska-praca-k6jwcdmpgsjuck4qyftlee.streamlit.app/

---

## 🎓 O projekte

**Segmify** je webová analytická aplikácia vytvorená pomocou frameworku **Streamlit**.  
Jej cieľom je transformovať surové transakčné dáta na prehľadné zákaznícke segmenty, ktoré môžu byť využité v marketingu, CRM a manažérskom reportingu.

Aplikácia umožňuje:

- načítanie a validáciu dát,
- čistenie a štandardizáciu datasetu,
- výpočet RFM metrík,
- segmentáciu zákazníkov pomocou clustering algoritmov,
- analýzu trendov v čase,
- generovanie marketingových odporúčaní,
- export výsledkov do CSV a ZIP.

---

## ✨ Hlavné funkcie

### 📂 Načítanie dát
- nahratie CSV súboru,
- automatická detekcia separátora,
- mapovanie stĺpcov na štandardný formát,
- podpora `amount` alebo výpočtu `Quantity × Price`,
- história uložených datasetov,
- možnosť pracovať aj s verejne dostupnými testovacími datasetmi.

**Online retail II** dataset bol zároveň použitý ako referenčný zdroj pri overovaní hypotéz a pri experimentálnom posudzovaní navrhnutého modelu segmentácie:
- [online_retail_II.zip](https://github.com/user-attachments/files/26783641/online_retail_II.zip)

  Na overenie funkčnosti aplikácie a testovanie jednotlivých analytických modulov je možné využiť aj verejne dostupné datasety z platformy **Kaggle**, napríklad:
- https://www.kaggle.com/datasets/umuttuygurr/e-commerce-customer-behavior-and-sales-analysis-tr
- https://www.kaggle.com/datasets/logiccraftbyhimanshi/walmart-customer-purchase-behavior-dataset
 
### 📊 RFM analýza
- výpočet metrík:
  - **Recency**
  - **Frequency**
  - **Monetary**
- kvantilové skórovanie zákazníkov,
- vážené RFM skóre,
- automatické priradenie segmentov.

### 🧠 Segmentácia zákazníkov
- clustering na základe RFM dát,
- podpora algoritmov:
  - **K-Means**
  - **DBSCAN**
  - **Hierarchical clustering**
- voľba vstupných premenných,
- škálovanie dát,
- auto-k analýza pomocou:
  - Elbow metódy,
  - Silhouette skóre.

### 📈 Trendy a monitoring
- sledovanie tržieb v čase,
- vývoj počtu zákazníkov,
- vývoj priemernej hodnoty objednávky,
- porovnanie segmentov,
- podiel segmentov na tržbách,
- identifikácia rastúcich a klesajúcich segmentov.

### 🎯 Marketingové odporúčania
- vysvetlenie logiky segmentácie,
- význam jednotlivých segmentov,
- odporúčané marketingové stratégie,
- export zákazníkov vybraného segmentu.

### 📄 Report a export
- súhrnné KPI ukazovatele,
- kontrola stavu analýzy,
- export:
  - transakcií,
  - RFM tabuľky,
  - segmentácie,
- ZIP export všetkých dostupných výstupov.

### ⚙️ Nastavenia
- úprava váh RFM metrík,
- výber predvoleného scaleru,
- nastavenie algoritmu segmentácie,
- konfigurácia rozsahu auto-k analýzy.

---

## 🧭 Štruktúra aplikácie

Aplikácia pozostáva z týchto stránok:

1. **Hlavná stránka**
2. **Načítanie dát**
3. **RFM analýza**
4. **Segmentácia zákazníkov**
5. **Trendy a monitoring**
6. **Marketingové odporúčania**
7. **Report a export**
8. **Nastavenia**

---

## 🗂️ Dátový model

Po načítaní a spracovaní sú dáta prevedené do jednotnej schémy:

| Stĺpec | Význam |
|---|---|
| `customer_id` | identifikátor zákazníka |
| `transaction_date` | dátum transakcie |
| `amount` | hodnota transakcie |

---

## 🧪 Odporúčaný postup používania

1. Nahrať transakčný dataset vo formáte CSV  
2. Namapovať stĺpce a uložiť dataset  
3. Spustiť **RFM analýzu**  
4. Spustiť **segmentáciu zákazníkov**  
5. Preskúmať **trendy a monitoring**  
6. Zobraziť **marketingové odporúčania**  
7. Vygenerovať **report a export**

---

## 🚀 Spustenie aplikácie

### Online verzia
Aplikácia je dostupná online na adrese:

```text
https://bakalarska-praca-k6jwcdmpgsjuck4qyftlee.streamlit.app/
