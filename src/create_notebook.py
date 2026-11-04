"""
Generates notebooks/exploratory_analysis.ipynb with clean structure, markdown explanations, and executable code.
"""

import json
import os

NB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "notebooks", "exploratory_analysis.ipynb")

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# STUDENT PERFORMANCE ANALYSIS AND PREDICTION USING EDA AND DEEP LEARNING\n",
            "### Course Code: U21ADP05 – Exploratory Data Analysis and Visualization\n",
            "**Academic Year:** 2026–2027  \n",
            "**Institution:** KPR Institute of Engineering and Technology  \n",
            "**Department:** Artificial Intelligence and Data Science  \n",
            "**Student Name:** Shreenidhi E.  \n\n",
            "---\n",
            "## 1. Project Overview\n",
            "This notebook presents the end-to-end exploratory data analysis, statistical hypothesis testing, and deep learning modeling for student final grade prediction using the official UCI Student Performance Dataset (`student-mat.csv`)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import sys\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "import tensorflow as tf\n\n",
            "# Add src directory to system path\n",
            "sys.path.append(os.path.abspath('../src'))\n",
            "from data_loader import load_dataset, get_dataset_summary\n",
            "from preprocessing import execute_full_preprocessing\n",
            "from eda import compute_univariate_statistics, compute_bivariate_analysis\n",
            "from statistical_analysis import analyze_all_numerical_distributions, perform_hypothesis_tests\n",
            "from visualizations import generate_all_visualizations\n",
            "from model import run_full_training_pipeline\n",
            "from evaluation import generate_all_evaluation_artifacts\n",
            "from prediction import GradePredictor\n\n",
            "print('Environment and libraries initialized successfully.')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Dataset Acquisition & Structural Validation\n",
            "Load the Mathematics cohort dataset (395 records, 33 attributes) from `dataset/student-mat.csv`."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df = load_dataset('../dataset/student-mat.csv')\n",
            "summary = get_dataset_summary(df)\n",
            "print(f\"Records: {summary['num_records']}, Attributes: {summary['num_attributes']}\")\n",
            "print(f\"Target column G3 present: {'G3' in summary['columns']}\")\n",
            "print(f\"Missing values: {summary['missing_values']}\")\n",
            "df.head()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Data Preprocessing & Leakage Prevention\n",
            "We split data into 70% Train (276), 15% Validation (59), and 15% Test (60) splits using `random_state=42`. All scalers and one-hot encoders are fitted strictly on the training partition."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "prep_res = execute_full_preprocessing(df, pipeline_path='../models/preprocessing_pipeline.pkl', cleaned_path='../dataset/student-mat-cleaned.csv')\n",
            "print(f\"Train shape: {prep_res['X_train_proc'].shape}\")\n",
            "print(f\"Validation shape: {prep_res['X_val_proc'].shape}\")\n",
            "print(f\"Test shape: {prep_res['X_test_proc'].shape}\")\n",
            "print(f\"Processed feature count: {prep_res['processed_feature_count']}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Exploratory Data Analysis & Statistical Testing\n",
            "Compute univariate statistics (mean, median, IQR, skewness, kurtosis) and run formal hypothesis tests."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "uni_stats = compute_univariate_statistics(df)\n",
            "display(uni_stats.head(10))\n\n",
            "dist_analysis = analyze_all_numerical_distributions(df)\n",
            "display(dist_analysis)\n\n",
            "hyp = perform_hypothesis_tests(df)\n",
            "for k, v in hyp.items():\n",
            "    print(f\"{k}: {v['Test']} -> p-value: {v['p_value']:.4e} | {v['Interpretation']}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Visualizations (8 Required Visualizations)\n",
            "Generate publication-quality charts for the 8 core requirements."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "plt.figure(figsize=(8, 4.5))\n",
            "sns.histplot(df['G3'], bins=21, kde=True, color='#1f77b4', edgecolor='black')\n",
            "plt.axvline(df['G3'].mean(), color='red', linestyle='--', label=f\"Mean: {df['G3'].mean():.2f}\")\n",
            "plt.axvline(10, color='orange', linestyle=':', label='Pass Threshold (10)')\n",
            "plt.title('Histogram of Student Final Grades (G3)')\n",
            "plt.xlabel('Final Grade (0 to 20)')\n",
            "plt.ylabel('Frequency')\n",
            "plt.legend()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Deep Learning Model Training & Hyperparameter Tuning\n",
            "Train Keras Multi-Layer Perceptron (MLP) regressor [128-64-32 with Dropout 0.2] and evaluate test set performance."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load evaluation artifacts generated from training pipeline\n",
            "eval_arts = generate_all_evaluation_artifacts()\n",
            "display(eval_arts['diagnostics']['comparison_table'])\n",
            "print('Overfitting:', eval_arts['diagnostics']['overfit_status'])\n",
            "print('Underfitting:', eval_arts['diagnostics']['underfit_status'])"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Interactive Prediction & Interventions\n",
            "Demonstrate real-time inference using the trained deep learning model and fitted pipeline."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "predictor = GradePredictor(pipeline_path='../models/preprocessing_pipeline.pkl', model_path='../models/best_model.keras')\n",
            "sample_student = {\n",
            "    'school': 'GP', 'sex': 'F', 'age': 17, 'address': 'U', 'famsize': 'GT3', 'Pstatus': 'T',\n",
            "    'Medu': 3, 'Fedu': 3, 'Mjob': 'other', 'Fjob': 'other', 'reason': 'course', 'guardian': 'mother',\n",
            "    'traveltime': 1, 'studytime': 3, 'failures': 0, 'schoolsup': 'no', 'famsup': 'yes', 'paid': 'no',\n",
            "    'activities': 'yes', 'nursery': 'yes', 'higher': 'yes', 'internet': 'yes', 'romantic': 'no',\n",
            "    'famrel': 4, 'freetime': 3, 'goout': 2, 'Dalc': 1, 'Walc': 1, 'health': 4, 'absences': 4,\n",
            "    'G1': 13, 'G2': 14\n",
            "}\n",
            "pred_out = predictor.predict(sample_student)\n",
            "print(f\"Predicted Final Grade: {pred_out['predicted_grade']} / 20 ({pred_out['tier']})\")\n",
            "print(f\"Summary: {pred_out['summary']}\")\n",
            "print('Recommendations:')\n",
            "for r in pred_out['recommendations']:\n",
            "    print(f\" - [{r['area']}] {r['text']}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Academic References (IEEE Style)\n",
            "[1] P. Cortez and A. Silva, 'Using data mining to predict secondary school student performance,' in *Proceedings of 5th Annual Future Business Technology Conference (FUBUTEC 2008)*, Porto, Portugal, 2008, pp. 5–12.  \n",
            "[2] D. Dua and C. Graff, 'UCI Machine Learning Repository: Student Performance Data Set,' University of California, Irvine, School of Information and Computer Sciences, 2017. [Online]. Available: https://archive.ics.uci.edu/dataset/320/student+performance.  \n",
            "[3] M. Abadi et al., 'TensorFlow: A system for large-scale machine learning,' in *12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16)*, 2016, pp. 265–283.  \n",
            "[4] F. Pedregosa et al., 'Scikit-learn: Machine learning in Python,' *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.  \n",
            "[5] C. Romero and S. Ventura, 'Educational data mining: A review of the state of the art,' *IEEE Transactions on Systems, Man, and Cybernetics, Part C (Applications and Reviews)*, vol. 40, no. 6, pp. 601–618, Nov. 2010."
        ]
    }
]

notebook_dict = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

os.makedirs(os.path.dirname(NB_PATH), exist_ok=True)
with open(NB_PATH, "w") as f:
    json.dump(notebook_dict, f, indent=2)

print(f"Jupyter Notebook successfully written to {NB_PATH}")
