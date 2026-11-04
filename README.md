# STUDENT PERFORMANCE ANALYSIS AND PREDICTION USING EXPLORATORY DATA ANALYSIS AND DEEP LEARNING



---

## 📌 Executive Summary & Problem Statement

Student academic achievement in secondary education is a composite outcome shaped by study habits, previous scholastic performance, familial background, institutional tutoring, and lifestyle choices. Identifying students at risk of underperformance early in the academic cycle enables institutions to deploy timely remedial interventions.

This project delivers an end-to-end data analytics and deep learning framework utilizing the benchmark **UCI Student Performance Dataset** (Mathematics cohort, 395 student instances, 33 attributes). Through comprehensive Exploratory Data Analysis (EDA), formal distribution modeling, and Ben Fry's Seven Stages of Data Visualization, we analyze key performance drivers and deploy a **Multi-Layer Perceptron (MLP)** regressor predicting final academic marks ($G3$) on a continuous scale of 0 to 20 points.

---

## 🚀 Key Highlights & Empirical Results

| Metric / Dimension | Baseline Linear Regression | Random Forest Regressor | Deep Learning MLP (Best Model) | Early Prediction MLP (No $G1/G2$) |
|:---|:---:|:---:|:---:|:---:|
| **Test MAE (Mean Absolute Error)** | 1.5164 | 1.1417 | **1.8189** | 3.1436 |
| **Test RMSE (Root Mean Sq Error)** | 2.0554 | 1.8695 | **2.3492** | 3.9916 |
| **Test $R^2$ Score (Variance Explained)** | 0.7891 | 0.8255 | **0.7245** | 0.2045 |
| **Overfitting Status** | Moderate collinearity | Low variance | **Controlled Regularization (Dropout 0.2)** | High bias |

- **Empirical Takeaway:** Historical academic momentum ($G1$ and $G2$) strongly accounts for final grade variance ($r > 0.80$). Without prior marks, early prediction achieves an MAE of 3.14, providing valuable early risk signals before semester exams.

---

## 📂 Project Directory Structure

```text
student_performance_analytics/
├── app.py                                # Master interactive Streamlit dashboard (11 sections)
├── student_performance_colab.ipynb       # 1-Click Master Google Colab Execution Notebook
├── requirements.txt                      # Project dependencies specification
├── README.md                             # Comprehensive project documentation
├── .gitignore                            # Standard git exclusion rules
│
├── dataset/
│   ├── student-mat.csv                   # Verified UCI Mathematics cohort dataset (395 x 33)
│   ├── student-mat-raw.csv               # Pristine raw UCI dataset backup
│   └── student-mat-cleaned.csv           # Cleaned dataset copy
│
├── models/
│   ├── best_model.keras                  # Trained Keras MLP regression model
│   └── preprocessing_pipeline.pkl        # Serialized scikit-learn preprocessing pipeline
│
├── src/
│   ├── data_loader.py                    # Automatic UCI download, extraction, and validation
│   ├── preprocessing.py                  # 70/15/15 train-val-test split & leakage prevention
│   ├── eda.py                            # Univariate, bivariate, and multivariate analysis
│   ├── statistical_analysis.py           # Normality tests, ANOVA, and hypothesis testing
│   ├── visualizations.py                 # 8 required publication charts generator
│   ├── model.py                          # Deep Learning MLP, tuning & baseline training
│   ├── evaluation.py                     # Metric calculation & diagnostic evaluation plots
│   ├── prediction.py                     # Inference engine & student intervention advisor
│   └── report_generator.py               # Automated PDF and Word academic report generator
│
├── pages/
│   ├── 1_Dataset_Overview.py             # Streamlit page: Attribute dictionary & raw data
│   ├── 2_Data_Preprocessing.py           # Streamlit page: Outlier & pipeline inspection
│   ├── 3_Exploratory_Analysis.py         # Streamlit page: Univariate & bivariate stats
│   ├── 4_Statistical_Analysis.py         # Streamlit page: Distribution tests & ANOVA
│   ├── 5_Visualizations.py               # Streamlit page: 8 assignment charts
│   ├── 6_Deep_Learning_Model.py          # Streamlit page: MLP architecture & tuning
│   ├── 7_Model_Evaluation.py             # Streamlit page: Test metrics & diagnostic plots
│   └── 8_Student_Prediction.py           # Streamlit page: Real-time grade prediction form
│
├── outputs/
│   ├── charts/                           # High-resolution PNG figures (DPI 300)
│   │   ├── 1_histogram_g3.png
│   │   ├── 2_boxplot_g3.png
│   │   ├── 3_barchart_studytime.png
│   │   ├── 4_scatterplot_absences.png
│   │   ├── 5_correlation_heatmap.png
│   │   ├── 6_pairplot_numerical.png
│   │   ├── 7_countplot_categorical.png
│   │   ├── 8_distribution_comparison.png
│   │   ├── eval_loss_curves.png
│   │   ├── eval_actual_vs_predicted.png
│   │   ├── eval_error_distribution.png
│   │   └── eval_baseline_comparison.png
│   ├── results/
│   │   └── model_results.json            # Actual empirical training & test metrics
│   └── reports/                          # Mirrored generated reports
│
├── notebooks/
│   ├── exploratory_analysis.ipynb        # Local Jupyter exploratory analysis notebook
│   └── student_performance_colab.ipynb   # Colab-ready notebook copy
│
└── report/
    ├── project_report.pdf                # Formal Academic PDF Report (ReportLab)
    └── project_report.docx               # Editable Word Document (python-docx)
```

---

## 🌐 Running in Google Colab (Step-by-Step)

You can run this entire project in **Google Colab** with zero local configuration.

### Option A: Uploading `student_performance_colab.ipynb` to Colab
1. Navigate to [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and select `student_performance_colab.ipynb` from this project folder.
3. In the Colab menu, select **Runtime > Change runtime type > Python 3 (GPU / T4 recommended)**.
4. Run each cell sequentially from top to bottom:
   - **Step 1:** Installs `reportlab`, `python-docx`, `streamlit`, `pyngrok`.
   - **Step 2:** Automatically downloads `student-mat.csv` from UCI.
   - **Step 3:** Runs leakage-free 70/15/15 train-val-test preprocessing.
   - **Step 4 & 5:** Executes EDA and renders all 8 high-resolution charts.
   - **Step 6 & 7:** Trains the Keras MLP model and displays evaluation metrics.
   - **Step 8:** Automatically compiles `project_report.pdf` and `project_report.docx` and downloads them directly to your computer.
   - **Step 9:** Launches the interactive Streamlit dashboard directly from Colab via localtunnel!

### Option B: Running the Streamlit App in Colab
Run the following commands inside a Colab code cell:
```python
!git clone <YOUR_GITHUB_REPO_URL>
%cd MAD
!pip install -r requirements.txt
!curl -s ipv4.icanhazip.com
!streamlit run app.py & npx localtunnel --port 8501
```
Click the generated `localtunnel` link and enter the IP address displayed by `icanhazip.com`.

---

## 💻 Local Installation & Execution

### 1. Prerequisites
- Python 3.10+ (Tested on Python 3.12)
- Git

### 2. Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd MAD

# Create and activate virtual environment (optional but recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Execution Commands
```bash
# 1. Download dataset & verify integrity
python src/data_loader.py

# 2. Run preprocessing & train Deep Learning model
python src/model.py

# 3. Generate all 8 publication visualizations & evaluation charts
python src/visualizations.py
python src/evaluation.py

# 4. Generate academic PDF and Word reports
python src/report_generator.py

# 5. Launch the Streamlit web dashboard
streamlit run app.py
```
The application will launch automatically in your browser at `http://localhost:8501`.

---

## 🔬 Exploratory Data Analysis & Statistical Modeling

### A. Univariate Diagnostics
- **Target Variable ($G3$):** Mean = 10.42, Median = 11.00, Variance = 20.99, Std = 4.58.
- **Skewness & Kurtosis:** Skewness = -0.732, Kurtosis = 0.400. Negative skewness is driven by a distinct cohort of 38 students scoring 0 points (dropouts/non-attendance).
- **Normality Test:** Shapiro-Wilk test on $G3$ yields $W = 0.928$ ($p = 1.34 \times 10^{-12}$), formally rejecting the hypothesis of normality.

### B. Hypothesis Testing & Bivariate Findings
1. **One-Way ANOVA (Study Time $\rightarrow G3$):**  
   $F = 3.65$, $p = 0.0128$ ($p < 0.05$). Weekly study time significantly elevates median grades ($<2$ hrs: 9.38 vs $5-10$ hrs: 11.41).
2. **Welch's Two-Sample t-test (Higher Education Desire):**  
   $t = 4.41$, $p = 0.00018$. Students aspiring for higher education average 10.68 vs 6.80 for non-aspirants (a 3.88 mark premium).
3. **Absence Correlation Paradox:**  
   Pearson $r = +0.034$ ($p = 0.505$). Absences do not exhibit an automatic linear penalty because top-performing students take occasional planned leaves, while several zero-grade students had 0 recorded absences due to early semester withdrawal.

---

## 🎨 Ben Fry's Seven Stages of Data Visualization

| Stage | Definition | Implementation in this System |
|:---|:---|:---|
| **1. ACQUIRE** | Ingest data from external repositories | Automatic HTTP streaming and extraction of UCI zip archive into `dataset/student-mat.csv`. |
| **2. PARSE** | Decode raw data and infer types | Semicolon parsing, classification into numeric/ordinal/nominal types, and 33-attribute schema dictionary. |
| **3. FILTER** | Remove noise and isolate subsets | Missing value checks, IQR outlier threshold calculation, and leakage-free 70/15/15 partitioning. |
| **4. MINE** | Statistical and pattern discovery | Univariate aggregates (mean, median, IQR), ANOVA tests, t-tests, and Pearson correlation matrices. |
| **5. REPRESENT** | Choose visual grammars | Histograms with KDE, boxplots with strip overlays, scatter with OLS trendlines, and heatmaps. |
| **6. REFINE** | Enhance clarity and engagement | High-contrast palettes, dual mean/median annotation lines, error bars, and axis labels. |
| **7. INTERACT** | Enable dynamic user manipulation | Interactive Plotly charts and Streamlit form controls for real-time scenario simulation. |

---

## 🧠 Deep Learning Architecture & Hyperparameters

```text
[Input Layer: 41 Transformed Features]
                 │
                 ▼
[Hidden Layer 1: Dense(128, ReLU)]
                 │
                 ▼
[Dropout Regularization (Rate = 0.2)]
                 │
                 ▼
[Hidden Layer 2: Dense(64, ReLU)]
                 │
                 ▼
[Dropout Regularization (Rate = 0.2)]
                 │
                 ▼
[Hidden Layer 3: Dense(32, ReLU)]
                 │
                 ▼
[Output Layer: Dense(1, Linear)] ──► Predicted Grade G3 (0 to 20 Scale)
```

- **Optimizer:** Adam (Learning Rate = 0.001)
- **Loss Function:** Mean Squared Error (MSE)
- **Evaluation Metric:** Mean Absolute Error (MAE)
- **Batch Size:** 32 | **Epochs:** 100 with Early Stopping (Patience = 15)
- **Model Checkpointing:** Optimal weights saved to `models/best_model.keras`

---
 

---


