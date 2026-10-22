# 🚢 Exploratory Data Analysis (EDA) on Titanic Dataset

**Internship Track:** Data Science & AI Internship  
**Project Number:** Project 1  
**Author:** Second-Year B.Tech Data Science & AI Intern  
**Technologies:** Python, Pandas, NumPy, Matplotlib, Seaborn, Jupyter Notebook  

---

## 📌 Project Overview & Objective

The objective of this project is to perform an end-to-end **Exploratory Data Analysis (EDA)** on the historical Titanic passenger dataset using Python and Pandas. 

The Titanic disaster is one of the most famous maritime tragedies in history. On April 15, 1912, the RMS Titanic sank after colliding with an iceberg, resulting in the death of 1,502 out of 2,224 passengers and crew. In this project, we analyze a representative sample of 891 passengers to identify the demographic, socioeconomic, and spatial factors that determined passenger survival.

### Key Objectives:
1. Load, inspect, and audit the Titanic dataset from an authoritative public repository.
2. Identify missing values, duplicates, and data types.
3. Apply statistically sound imputation techniques for numerical and categorical missing values with clear justifications.
4. Perform exploratory grouping and aggregations across gender, passenger class, and embarkation ports.
5. Create publication-quality data visualizations saved in the `outputs/` folder.
6. Derive, calculate, and document verified statistical findings that tell the story of the data.
7. Export the clean dataset to `cleaned_titanic.csv`.

---

## 🛠️ Technologies & Tools

- **Programming Language:** Python 3.11+ / 3.14+
- **Data Manipulation & Analysis:** `pandas` (>= 2.0.0), `numpy` (>= 1.24.0)
- **Data Visualization:** `matplotlib` (>= 3.7.0), `seaborn` (>= 0.12.0)
- **Interactive Notebooks:** `jupyter`

---

## 📁 Repository Structure

```
Titanic_EDA/
│
├── eda_titanic.py              # Complete, production-grade Python script with beginner-friendly comments
├── Titanic_EDA.ipynb           # Interactive Jupyter Notebook version
├── requirements.txt            # Python dependencies
├── titanic.csv                 # Raw dataset (891 rows, 12 columns)
├── cleaned_titanic.csv         # Cleaned, imputed dataset (891 rows, 12 columns, 0 missing values)
│
├── outputs/                    # Exported high-resolution (300 DPI) visual charts
│   ├── 01_survival_count.png       # Passenger survival vs mortality counts
│   ├── 02_age_distribution.png     # Age distribution histogram with KDE & median line
│   ├── 03_survival_by_pclass.png   # Survival rate by passenger class (1st, 2nd, 3rd)
│   ├── 04_survival_by_gender.png   # Survival rate comparison between male & female
│   └── 05_correlation_heatmap.png  # Correlation matrix of numerical features
│
└── README.md                   # Comprehensive project documentation & viva defense guide
```

---

## 🔍 Dataset Description

The dataset consists of **891 rows** (passengers) and **12 columns** (features).

| Column | Full Name | Type | Description | Values / Range |
| :--- | :--- | :--- | :--- | :--- |
| `PassengerId` | Passenger Identifier | Integer | Unique identifier for each passenger | 1 to 891 |
| `Survived` | Survival Status | Integer | Target variable indicating survival | `0` = Perished, `1` = Survived |
| `Pclass` | Passenger Ticket Class | Integer | Socioeconomic proxy | `1` = 1st (Upper), `2` = 2nd (Middle), `3` = 3rd (Lower) |
| `Name` | Passenger Full Name | String | Name including honorific titles (Mr, Mrs, Miss, Master, etc.) | Text string |
| `Sex` | Biological Gender | String | Passenger gender | `'male'`, `'female'` |
| `Age` | Passenger Age | Float | Age in fractional or whole years | 0.42 to 80.0 years |
| `SibSp` | Siblings / Spouses | Integer | Count of siblings or spouses aboard | 0 to 8 |
| `Parch` | Parents / Children | Integer | Count of parents or children aboard | 0 to 6 |
| `Ticket` | Ticket Code | String | Ticket number/string | Alpha-numeric |
| `Fare` | Ticket Price | Float | Passenger fare paid in British pounds | £0.00 to £512.33 |
| `Cabin` | Cabin Number | String | Stateroom / berth identifier | Deck letter + number (e.g., C85) |
| `Embarked` | Port of Embarkation | String | Port where passenger boarded Titanic | `'C'` = Cherbourg, `'Q'` = Queenstown, `'S'` = Southampton |

---

## 🧼 Data Cleaning & Imputation Methodology

### 1. Duplicate Records Audit
- **Check Performed:** `df.duplicated().sum()`
- **Result:** **0 duplicate rows detected.**
- **Action Taken:** No rows removed. Every passenger ID and record represents a distinct passenger entry.

### 2. Missing Value Treatment & Statistical Justifications

Auditing the raw dataset revealed missing values in 3 columns:
- **`Age`**: 177 missing values (19.87%)
- **`Cabin`**: 687 missing values (77.10%)
- **`Embarked`**: 2 missing values (0.22%)

```
Column        Missing Count    Percentage (%)    Selected Handling Strategy
----------------------------------------------------------------------------------------
Age           177              19.87%            Median Imputation (28.0 years)
Embarked      2                0.22%             Mode Imputation ('S' / Southampton)
Cabin         687              77.10%            Categorical Imputation ('Unknown')
```

#### Detailed Justifications:

1. **`Age` (Numerical) &rarr; Median Imputation (`Age = 28.0 years`):**
   - *Why not Mean?* The age distribution is right-skewed with extreme boundaries: infants under 1 year (0.42 years) up to 80-year-olds. The arithmetic mean ($\approx 29.70$ years) is sensitive to extreme values. The median ($\approx 28.0$ years) represents the true 50th percentile and is robust against outlier distortion.
   - *Result:* Preserved all 891 passenger records without altering the central tendency.

2. **`Embarked` (Categorical) &rarr; Mode Imputation (`Embarked = 'S'`):**
   - *Why Mode?* Only 2 records out of 891 (0.22%) were missing. Over 72.4% (644 out of 889 known passengers) embarked at Southampton (`S`).
   - *Result:* Assigning the two passengers to the dominant port maintains distribution stability with negligible statistical bias.

3. **`Cabin` (Categorical) &rarr; Impute with `'Unknown'`:**
   - *Why not drop rows?* Dropping rows with missing cabins would discard 687 rows (**77.1% of the dataset**), severely shrinking the sample size and inducing catastrophic selection bias.
   - *Why not drop the column?* Cabin presence itself is an informative indicator of wealth: 1st class passengers had recorded cabins, whereas lower-deck passengers rarely had staterooms recorded.
   - *Result:* Replacing `NaN` with `'Unknown'` safely preserves 100% of the dataset and retains the unassigned cabin signal.

---

## 📊 Grouping, Aggregation & Real Statistical Findings

All calculations are computed directly from the actual dataset:

### 1. Survival Rate by Gender
```
Gender     Total Passengers    Survivors    Survival Rate (%)
-------------------------------------------------------------
Female     314                 233          74.20%
Male       577                 109          18.89%
```
- **Finding:** Females had a **74.20%** survival rate compared to **18.89%** for males. A female passenger was approximately **3.9 times more likely** to survive than a male passenger.
- **Real-World Context:** This provides empirical statistical confirmation of Captain Edward Smith's strict enforcement of the maritime evacuation protocol: *"Women and children first"*.

### 2. Survival Rate by Passenger Ticket Class (Pclass)
```
Class        Total Passengers    Survivors    Survival Rate (%)
-------------------------------------------------------------
1st Class    216                 136          62.96%
2nd Class    184                  87          47.28%
3rd Class    491                 119          24.24%
```
- **Finding:** First-class passengers enjoyed a **62.96%** survival rate, second-class experienced **47.28%**, while third-class passengers experienced only **24.24%**.
- **Real-World Context:** Survival probability was directly proportional to socioeconomic status. First-class cabins were situated on upper decks adjacent to the lifeboat deck, whereas third-class passengers were housed deep in lower compartments behind gated stairwells.

### 3. Average & Median Age by Passenger Class
```
Class        Mean Age (Years)    Median Age (Years)    Min Age    Max Age
-------------------------------------------------------------------------
1st Class    36.81               35.0                  0.92       80.0
2nd Class    29.77               28.0                  0.67       66.0
3rd Class    25.93               28.0                  0.42       74.0
```
- **Finding:** The average age demonstrates a clear downward gradient: 1st Class (36.81 years) > 2nd Class (29.77 years) > 3rd Class (25.93 years).
- **Real-World Context:** First-class travelers were older, established aristocrats, industrialists, and executives who had accumulated wealth. Third-class passengers were predominantly young, working-class immigrants traveling to North America for labor opportunities.

### 4. Passenger Distribution and Survival by Embarkation Port
```
Port Code    Port Name      Passengers    Share (%)    Survival Rate (%)
-------------------------------------------------------------------------
S            Southampton    646           72.50%       33.90%
C            Cherbourg      168           18.86%       55.36%
Q            Queenstown      77            8.64%       38.96%
```
- **Finding:** While Southampton was the primary embarkation port (72.50% of all passengers), Cherbourg passengers had a dramatically higher survival rate (**55.36%** vs 33.90% at Southampton).
- **Underlying Cause:** Over **50.6%** of travelers boarding at Cherbourg booked luxury 1st-class tickets. The higher survival rate was driven by the socioeconomic profile of passengers boarding in France rather than the port itself.

### 5. Overall Survival vs. Mortality Summary
- **Total Passengers in Sample:** 891
- **Total Perished:** 549 (**61.62%**)
- **Total Survived:** 342 (**38.38%**)
- **Finding:** More than 6 out of 10 passengers perished, underscoring the tragic reality of insufficient lifeboats (Titanic carried lifeboats for only ~1,178 people despite carrying over 2,200 onboard).

---

## 📈 Visualizations Gallery

The project automatically generates and saves 5 high-resolution charts in `outputs/`:

1. **`01_survival_count.png`**: Count plot comparing passengers who perished ($549$, $61.6\%$) versus survived ($342$, $38.4\%$) with explicit count and percentage labels.
2. **`02_age_distribution.png`**: Histogram with Kernel Density Estimation (KDE) curve illustrating the passenger age distribution, including marked dashed lines for median age ($28.0$ years) and mean age ($29.7$ years).
3. **`03_survival_by_pclass.png`**: Bar chart detailing the survival rate drop across 1st Class ($62.96\%$), 2nd Class ($47.28\%$), and 3rd Class ($24.24\%$).
4. **`04_survival_by_gender.png`**: Comparative bar plot highlighting the stark contrast in survival between Females ($74.20\%$) and Males ($18.89\%$).
5. **`05_correlation_heatmap.png`**: Annotated heatmap displaying Pearson correlation coefficients across numerical attributes (`Survived`, `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`), demonstrating negative correlation between `Pclass` and `Survived` ($-0.34$) and positive correlation between `Fare` and `Survived` ($+0.26$).

---

## 🚀 How to Run the Project

### Prerequisites
Ensure Python is installed on your system (Python 3.10, 3.11, 3.12, 3.13, or 3.14).

### Step 1: Open Terminal / PowerShell
Navigate to the project directory:
```powershell
cd c:\Users\polla\OneDrive\Desktop\Titanic_EDA
```

### Step 2: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Execute the Python Script
Run the automated analysis script:
```powershell
python eda_titanic.py
```
*(On Windows with multiple Python versions installed, you can also use `py -3.11 eda_titanic.py`)*

### Step 4: (Optional) Run the Interactive Jupyter Notebook
```powershell
jupyter notebook Titanic_EDA.ipynb
```

---

## 🎓 Internship Review & Viva Defense Guide

As a 2nd-year B.Tech student, internship mentors frequently ask conceptual questions during project evaluations. Here are model answers based directly on this project:

#### Q1: "Why did you impute missing Age values with the Median instead of the Mean?"
> **Answer:** *"Age in this dataset exhibits positive skewness with extreme values ranging from infants under 1 year to an 80-year-old passenger. The arithmetic mean is sensitive to outliers and gets pulled towards the tail, whereas the median represents the true 50th percentile (robust central tendency). Imputing with the median (28.0 years) preserves the distribution's integrity without artificially inflating variance."*

#### Q2: "Why didn't you drop the `Cabin` column if 77% of its values were missing?"
> **Answer:** *"In production data science, dropping 77% of rows would eliminate over three-quarters of our observations. Dropping the column entirely would discard valuable signal: the presence of a recorded cabin is strongly correlated with 1st-class ticket holders and deck level. By imputing with the category `'Unknown'`, we retain 100% of the sample records while explicitly capturing the missingness pattern."*

#### Q3: "Did you find any duplicate records, and how did you verify them?"
> **Answer:** *"We performed a full duplicate check using `df.duplicated().sum()`, which returned 0 duplicates. We verified that every row possesses a unique `PassengerId` from 1 to 891. Therefore, no rows were removed, avoiding accidental loss of valid data."*

#### Q4: "What is the single most predictive feature for survival on the Titanic?"
> **Answer:** *"Biological sex (`Sex`) is the single strongest univariate predictor. Female passengers had a 74.20% survival rate versus 18.89% for males (a ~3.9x ratio), reflecting the strict 'women and children first' maritime protocol. Socioeconomic status (`Pclass`) is the second strongest factor, with 1st-class passengers surviving at more than 2.5x the rate of 3rd-class passengers."*

---

## 📋 Internship Submission Checklist

- [x] Dataset loaded from reliable public repository (`datasciencedojo/datasets`).
- [x] First 5 rows inspected using `.head()`.
- [x] Dataset information displayed using `.info()`.
- [x] Descriptive statistics generated using `.describe()`.
- [x] Shape (rows and columns) calculated and verified ($891 \times 12$).
- [x] Missing values identified across all columns (`isnull().sum()`).
- [x] Appropriate imputation applied (Median for Age, Mode for Embarked, 'Unknown' for Cabin) with written justifications.
- [x] Duplicate check verified ($0$ duplicates).
- [x] Groupings and aggregations computed for Gender, Pclass, Age, and Embarked.
- [x] Five visual charts exported to `outputs/` folder at 300 DPI.
- [x] Five verified statistical findings documented with exact numbers.
- [x] Clean dataset exported as `cleaned_titanic.csv`.
- [x] Comprehensive `README.md` and `Titanic_EDA.ipynb` included.
- [x] Clean code with beginner-friendly comments executed without errors.
