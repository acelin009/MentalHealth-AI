# 🧠 MentalHealth-AI

### Mental Health in the Technology Industry: Data Analysis, Machine Learning & Model Interpretability

An end-to-end Data Science project analyzing mental health trends in the technology industry using multi-year OSMI survey data.

The project combines surveys spanning multiple years into a harmonized dataset to explore how personal, demographic, and workplace factors relate to mental health treatment-seeking behavior.

It covers the complete Data Science lifecycle:

**Data Collection → Data Cleaning → Data Harmonization → Exploratory Data Analysis → Statistical Analysis → Feature Engineering → Machine Learning → Hyperparameter Tuning → Model Interpretability**

---

## 📌 Project Overview

Mental health has become an increasingly important concern in the technology industry.

Employees may experience workplace stress, stigma surrounding mental health, limited organizational support, and uncertainty about accessing professional treatment.

This project investigates patterns in mental health survey responses from technology workers and attempts to answer questions such as:

- How have mental health attitudes changed across survey years?
- How common is professional mental health treatment among respondents?
- Does workplace support influence treatment-seeking behavior?
- Are employees aware of the mental health resources available to them?
- How do workplace benefits and organizational policies differ across companies?
- Which factors are most strongly associated with seeking mental health treatment?
- Can machine learning predict whether an individual has sought professional treatment?

The final machine learning system predicts the target:

> **`sought_treatment`**

using demographic, personal, and workplace-related characteristics.

---

# 🎯 Project Objectives

The main objectives of this project are:

1. Combine multiple OSMI mental health survey datasets.
2. Clean and harmonize inconsistent survey schemas across years.
3. Perform comprehensive exploratory data analysis.
4. Investigate demographic and workplace mental health patterns.
5. Analyze changes in mental health attitudes over time.
6. Perform statistical analysis to validate observed relationships.
7. Engineer meaningful features for machine learning.
8. Build a reproducible preprocessing and modeling pipeline.
9. Compare multiple classification algorithms.
10. Optimize the strongest model using hyperparameter tuning.
11. Interpret the final model using feature and permutation importance.
12. Translate model findings into meaningful business insights.

---

# 📊 Dataset

This project uses survey data from the **Open Sourcing Mental Illness (OSMI)** mental health surveys.

The datasets contain responses related to:

- Demographics
- Employment characteristics
- Company size
- Mental health benefits
- Workplace resources
- Mental health discussions
- Supervisor and coworker support
- Workplace anonymity
- Mental health history
- Family history
- Mental health treatment
- Workplace attitudes
- Mental health interference
- Organizational support

Because survey questions changed across years, combining the datasets required extensive schema harmonization and missing-value analysis.

---

# 🔄 Project Workflow

```text
Raw Multi-Year Survey Data
        │
        ▼
Data Understanding
        │
        ▼
Data Cleaning
        │
        ▼
Schema Harmonization
        │
        ▼
Exploratory Data Analysis
        │
        ├── Demographic Analysis
        ├── Workplace Analysis
        ├── Mental Health Analysis
        └── Time Trend Analysis
        │
        ▼
Statistical Analysis
        │
        ▼
Feature Engineering
        │
        ▼
Data Preprocessing
        │
        ├── Missing Value Strategy
        ├── Numerical Processing
        ├── Categorical Encoding
        └── Train/Test Split
        │
        ▼
Machine Learning
        │
        ├── Logistic Regression
        ├── Decision Tree
        └── Random Forest
        │
        ▼
Hyperparameter Tuning
        │
        ▼
Model Evaluation
        │
        ▼
Model Interpretability
        │
        ├── Feature Importance
        └── Permutation Importance
        │
        ▼
Business Insights
```

---

# 🧹 Data Cleaning & Harmonization

Combining multiple survey years introduced several real-world data quality challenges.

The cleaning pipeline addressed:

### Missing Values

Missingness was analyzed dynamically for every feature.

Columns with excessive missing values were evaluated and removed when insufficient information was available for reliable modeling.

Structural missingness caused by questions not appearing in particular survey years was treated separately from valid responses such as `"Don't Know"`.

### Inconsistent Binary Values

Different representations such as:

```text
True / False
Yes / No
1 / 0
1.0 / 0.0
```

were standardized.

### Demographic Harmonization

Gender values were consolidated into consistent categories.

Country names and other demographic variables were standardized to reduce duplicate categories caused by spelling and formatting differences.

### Company Size Harmonization

Company-size responses varied between survey years and required normalization into consistent categories.

### Categorical Standardization

Variations such as:

```text
Don't Know
Don't know
I don't know
```

were standardized while preserving meaningful uncertainty as a valid response.

### Data Type Optimization

Columns were converted to appropriate numerical and categorical data types to improve consistency and memory efficiency.

---

# 🔍 Exploratory Data Analysis

The EDA phase was divided into several focused analyses.

## 📊 Executive Overview

Provides a high-level understanding of:

- Dataset size
- Survey year distribution
- Feature composition
- Missing data patterns
- Overall treatment-seeking behavior

## 👥 Demographic Analysis

Explores patterns across:

- Age
- Gender
- Country
- Race
- Geographic distribution

## 🏢 Workplace Analysis

Investigates:

- Company size
- Mental health benefits
- Mental health resources
- Workplace anonymity
- Supervisor comfort
- Coworker comfort
- Workplace discussions
- Leave accessibility

## 🧠 Mental Health Analysis

Examines:

- Treatment-seeking behavior
- Family history
- Current mental health conditions
- Previous mental health conditions
- Diagnosed disorders
- Mental health interference

## 📈 Time Trend Analysis

Analyzes how responses and workplace attitudes changed across survey years.

---

# 📐 Statistical Analysis

Statistical analysis was performed to determine whether observed relationships were statistically meaningful.

The analysis explored associations between treatment-seeking behavior and selected demographic, workplace, and mental health variables.

This phase helped distinguish meaningful patterns from relationships that may have occurred by chance.

---

# 🛠 Feature Engineering

Several features were prepared and engineered to improve model usability.

The feature engineering process included:

- Removing redundant variables
- Handling high-cardinality categories
- Harmonizing company-size categories
- Preserving structural missingness
- Preparing categorical variables for encoding
- Identifying potential target leakage
- Separating numerical, categorical, and ordinal variables

Special care was taken to avoid blindly one-hot encoding free-text fields or extremely high-cardinality variables.

---

# 🤖 Machine Learning

The machine learning task was formulated as a binary classification problem.

### Target Variable

```text
sought_treatment
```

The goal is to predict whether a survey respondent has sought professional mental health treatment.

Three baseline models were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

---

# 📊 Baseline Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 79.90% | 82.11% | 81.63% | 81.87% |
| Decision Tree | 72.60% | 75.14% | 75.80% | 75.47% |
| **Random Forest** | **80.39%** | 81.53% | **83.55%** | **82.74%** |

The **Random Forest classifier** was selected as the strongest baseline model because it achieved:

- The highest recall
- The highest F1 score
- Competitive overall accuracy

For this problem, recall is particularly useful because false negatives represent respondents who sought treatment but were incorrectly classified as not having done so.

---

# ⚙️ Hyperparameter Tuning

The Random Forest model was further optimized using cross-validation and hyperparameter search.

Parameters explored included:

- Number of estimators
- Maximum tree depth
- Minimum samples required to split
- Minimum samples per leaf
- Number of features considered at each split

The best-performing configuration was selected as the final model.

---

# 🔬 Model Interpretability

Model interpretation was performed to understand which variables contributed most strongly to predictions.

The project uses:

### Random Forest Feature Importance

Measures how strongly transformed features contribute to decision-making across the ensemble of trees.

### Permutation Importance

Measures how much model performance decreases when individual input features are randomly shuffled.

Using multiple interpretability techniques provides a more complete understanding of model behavior.

---

# 💡 Key Insights

The analysis indicates that treatment-seeking behavior is influenced by a combination of personal mental health history and workplace conditions.

Important themes observed during the analysis include:

- Workplace mental health support varies significantly across organizations.
- Access to mental health benefits and resources is inconsistent.
- Employees differ considerably in their comfort discussing mental health with supervisors and coworkers.
- Family and personal mental health history are strongly related to treatment behavior.
- Organizational policies and workplace culture may influence willingness to seek professional support.
- Workplace-related factors can provide useful predictive information alongside personal characteristics.

These findings highlight the importance of creating supportive workplace environments where employees can access mental health resources without fear of stigma.

---

# ⚠️ Ethical Considerations

Mental health data is highly sensitive.

This project is intended for:

- Educational purposes
- Exploratory data analysis
- Machine learning research
- Portfolio demonstration

The predictive model should **not** be used to:

- Diagnose mental health conditions
- Replace professional medical advice
- Make employment decisions
- Screen employees for mental health conditions
- Discriminate against individuals based on predicted outcomes

The model predicts patterns associated with **treatment-seeking behavior**, not whether an individual has a mental illness.

Predictions should therefore be interpreted carefully and responsibly.

---

# ⚠️ Project Limitations

Several limitations should be considered when interpreting the results.

### Survey Population

The dataset primarily represents individuals working in or associated with the technology industry.

Results may not generalize to the broader population.

### Self-Reported Data

Survey responses are self-reported and may contain recall bias or response bias.

### Changing Survey Questions

Different OSMI survey years contain different questions.

This creates structural missingness and makes direct year-to-year comparisons challenging.

### Geographic Imbalance

Some countries have significantly more respondents than others.

### Prediction vs. Causation

Machine learning feature importance indicates predictive relationships, not causal relationships.

A feature being important to the model does not mean that it causes treatment-seeking behavior.

---

# 📁 Project Structure

```text
MentalHealth-AI/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_Data_Understanding.ipynb
│   ├── 02_Data_Cleaning.ipynb
│   ├── 03_Data_Harmonization.ipynb
│   ├── 04_01_Executive_Overview.ipynb
│   ├── 04_02_Demographic_Analysis.ipynb
│   ├── 04_03_Workplace_Analysis.ipynb
│   ├── 04_04_Mental_Health_Analysis.ipynb
│   ├── 04_05_Time_Trends.ipynb
│   ├── 04_06_Statistical_Analysis.ipynb
│   ├── 05_Feature_Engineering.ipynb
│   ├── 06_Data_Preprocessing.ipynb
│   ├── 07_Machine_Learning.ipynb
│   └── 08_Model_Interpretability.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explain.py
│   ├── predict.py
│   └── utils.py
│
├── models/
│   ├── best_model.pkl
│   ├── preprocessor.pkl
│   └── model_metadata.json
│
├── results/
│
├── tests/
│
├── .github/
│   └── workflows/
│
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 💻 Technologies Used

### Programming

- Python

### Data Analysis

- Pandas
- NumPy

### Data Visualization

- Matplotlib
- Plotly

### Machine Learning

- Scikit-learn

### Model Persistence

- Joblib

### Development

- Jupyter Notebook
- Git
- GitHub

### CI/CD

- GitHub Actions

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone https://github.com/acelin009/MentalHealth-AI.git
cd MentalHealth-AI
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run Jupyter Notebook

```bash
jupyter notebook
```

The notebooks can then be explored in numerical order to follow the complete project workflow.

---

# 🔁 Reproducibility

The project separates exploratory notebooks from reusable Python modules.

Reusable functionality is located inside:

```text
src/
```

This includes:

- Data loading
- Feature engineering
- Preprocessing
- Model training
- Model evaluation
- Prediction
- Model interpretation utilities

This structure reduces duplicated code and improves maintainability and reproducibility.

---

# 🔮 Future Improvements

Potential future extensions include:

- Testing gradient boosting models such as XGBoost or LightGBM
- Adding more advanced model interpretation methods
- Performing NLP analysis on open-ended survey responses
- Topic modeling of employee comments
- Developing a REST API for model inference
- Creating an optional interactive analytics dashboard
- Expanding the dataset with additional mental health surveys
- Investigating causal relationships between workplace policies and treatment behavior

---

# 📜 License

This project is intended for educational and portfolio purposes.

Please refer to the original dataset provider for dataset-specific licensing and usage requirements.

---

# 👤 Author

**Acelin Nazareth**

Bachelor of Engineering — Artificial Intelligence & Data Science

Interested in Data Science, Machine Learning, Data Analytics, and building data-driven solutions.

---

## ⭐ Acknowledgements

This project uses mental health survey data published by the Open Sourcing Mental Illness (OSMI) community.

Special thanks to the organizations and survey participants who contributed data toward improving understanding of mental health in the technology industry.

---

## ⭐ Support

If you found this project useful or interesting, consider giving the repository a ⭐.

Feedback, suggestions, and contributions are welcome.
