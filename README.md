Kod w Pythonie stanowi **kompleksowy pipeline analizy danych HR dotyczących rotacji pracowników (ang. *attrition*)**, który obejmuje **załadowanie danych, czyszczenie, eksplorację, inżynierię cech, modelowanie oraz ocenę modeli predykcyjnych**.

## 🧩 **Cel analizy**

Celem jest **predykcja odejścia pracownika z firmy (zmienna `Attrition`)**. Problem ten jest klasyfikacją binarną:

* `Attrition = Yes` – pracownik odszedł,
* `Attrition = No` – pracownik pozostał.

## 📦 **Etapy pipeline’u**

### 1. **Import bibliotek**

Importowane są biblioteki do przetwarzania danych (`numpy`, `pandas`), wizualizacji (`matplotlib`, `seaborn`) oraz modelowania (`sklearn`, `SMOTE` itd.).

### 2. **Wczytanie i scalanie danych**

```python
df1 = pd.read_csv("df1(1).csv")
df2 = pd.read_csv("df2(1).csv")
df = pd.merge(df1, df2, on="EmployeeNumber")
```

Dwa zbiory danych są scalane na podstawie identyfikatora pracownika (`EmployeeNumber`).

### 3. **Eksploracja danych**

* Histogramy (`df.hist()`)
* Unikalne wartości w kolumnach
* Liczba duplikatów i braków danych
* Korelacje między zmiennymi (`sns.heatmap`)
* Statystyki opisowe

### 4. **Czyszczenie danych**

* Usunięcie duplikatów i braków
* Usunięcie zbędnych kolumn (np. `Over18`, `EmployeeCount`)
* Konwersja typów danych (`int`, `category`)
* Filtracja nienaturalnych wartości (np. `Age < 100`)
* Walidacja logiczna (np. `YearsInCurrentRole < Age`)

### 5. **Inżynieria cech**

* Tworzenie zmiennych zero-jedynkowych z cech kategorycznych (`pd.get_dummies`)
* Usunięcie cech o wysokiej korelacji (`MonthlyIncome`, `YearlyIncome`)

### 6. **Analiza korelacji**

* Korelacja cech z `Attrition_Yes`
* Macierz korelacji
* Wyznaczanie cech o najwyższej bezwzględnej korelacji (`get_top_abs_correlations`)

### 7. **Podział na zbiór treningowy/testowy + standaryzacja**

```python
X_train, X_test, y_train, y_test = train_test_split(...)
StandardScaler()
```

### 8. **Modelowanie i ocena modeli**

#### 📌 Modele zastosowane:

* **Regresja logistyczna (L1, z i bez SMOTE)**
* **Random Forest (z tunowaniem hiperparametrów via `GridSearchCV`)**
* **Naive Bayes (z i bez SMOTE)**

#### 📊 Metryki i narzędzia oceny:

* Macierze pomyłek
* Accuracy, precision, recall, F1
* ROC curve, PR curve

## 🎯 Główne problemy rozwiązane przez kod

1. **Zbalansowanie nierównych klas** (odejście vs zostanie) przy pomocy **SMOTE**.
2. **Wybór i interpretacja ważnych cech wpływających na odejście pracownika**.
3. **Ocena modeli klasyfikacyjnych** i porównanie ich skuteczności.
4. **Automatyzacja procesu przetwarzania danych HR** pod kątem ML.

## ✅ Wnioski

Kod zawiera pełny proces:

* Od **czyszczenia danych** → **eksploracji** → **transformacji cech**
* Po **budowę i ocenę modeli predykcyjnych**
* Zawiera też **mechanizmy walidacji poprawności danych**, np. sprawdzanie logicznych niespójności typu `YearsInCurrentRole >= Age`.
