"""
Exploratory Data Analysis (EDA) Module for Student Performance Analytics.
Provides:
- Complete Univariate Statistical Summaries (Mean, Median, Mode, Variance, Std, Min, Max, Quartiles, Skewness, Kurtosis)
- Categorical Frequency Distributions
- Discrete vs Continuous Variable Characterization
- Bivariate Relationship Calculations
- Multivariate Correlation Matrix & Pairwise Metrics
"""

import numpy as np
import pandas as pd
from scipy import stats


def compute_univariate_statistics(df, columns=None):
    """
    Computes complete statistical profile for numerical attributes:
    Mean, Median, Mode, Variance, Standard Deviation, Min, Max,
    Q1 (25%), Q2 (50%), Q3 (75%), IQR, Skewness, Kurtosis.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    records = []
    for col in columns:
        series = df[col].dropna()
        mode_val = series.mode().iloc[0] if not series.mode().empty else np.nan
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        records.append({
            "Feature": col,
            "Count": int(len(series)),
            "Mean": round(float(series.mean()), 3),
            "Median": round(float(series.median()), 3),
            "Mode": round(float(mode_val), 3),
            "Variance": round(float(series.var()), 3),
            "Std_Dev": round(float(series.std()), 3),
            "Min": round(float(series.min()), 3),
            "Q1": round(float(q1), 3),
            "Q2_Median": round(float(series.median()), 3),
            "Q3": round(float(q3), 3),
            "IQR": round(float(iqr), 3),
            "Max": round(float(series.max()), 3),
            "Skewness": round(float(series.skew()), 3),
            "Kurtosis": round(float(series.kurtosis()), 3)
        })

    return pd.DataFrame(records)


def compute_categorical_distributions(df, columns=None):
    """
    Computes frequency and percentage distributions for categorical attributes.
    """
    if columns is None:
        columns = df.select_dtypes(include=["object"]).columns.tolist()

    dist_dict = {}
    for col in columns:
        counts = df[col].value_counts()
        percentages = df[col].value_counts(normalize=True) * 100
        summary_df = pd.DataFrame({
            "Category": counts.index,
            "Frequency": counts.values,
            "Percentage (%)": percentages.values.round(2)
        })
        dist_dict[col] = summary_df

    return dist_dict


def compute_bivariate_analysis(df):
    """
    Performs focused bivariate analysis between key independent features and final grade (G3):
    1. Study time vs G3
    2. Absences vs G3
    3. Previous grades (G1, G2) vs G3
    4. Family education (Medu, Fedu) vs G3
    5. Internet access vs G3
    6. School support vs G3
    """
    bivariate_summary = {}

    # 1. Study time vs G3
    study_grp = df.groupby("studytime")["G3"].agg(["count", "mean", "median", "std"]).round(2).reset_index()
    study_grp["studytime_label"] = study_grp["studytime"].map({1: "<2 hours", 2: "2-5 hours", 3: "5-10 hours", 4: ">10 hours"})
    bivariate_summary["studytime_vs_G3"] = study_grp

    # 2. Absences vs G3 (Correlation + Grouped Binned Analysis)
    abs_bins = [0, 2, 5, 10, 20, 100]
    abs_labels = ["0-2 (Very Low)", "3-5 (Low)", "6-10 (Moderate)", "11-20 (High)", ">20 (Severe)"]
    df_temp = df.copy()
    df_temp["absence_bin"] = pd.cut(df_temp["absences"], bins=abs_bins, labels=abs_labels, include_lowest=True)
    abs_grp = df_temp.groupby("absence_bin", observed=False)["G3"].agg(["count", "mean", "median", "std"]).round(2).reset_index()
    corr_abs_g3 = stats.pearsonr(df["absences"], df["G3"])
    bivariate_summary["absences_vs_G3"] = {
        "grouped": abs_grp,
        "pearson_r": round(float(corr_abs_g3.statistic), 3),
        "p_value": float(corr_abs_g3.pvalue)
    }

    # 3. Previous grades (G1, G2) vs G3
    corr_g1_g3 = stats.pearsonr(df["G1"], df["G3"])
    corr_g2_g3 = stats.pearsonr(df["G2"], df["G3"])
    bivariate_summary["previous_grades_vs_G3"] = {
        "G1_r": round(float(corr_g1_g3.statistic), 3),
        "G1_p": float(corr_g1_g3.pvalue),
        "G2_r": round(float(corr_g2_g3.statistic), 3),
        "G2_p": float(corr_g2_g3.pvalue)
    }

    # 4. Family education (Medu, Fedu) vs G3
    medu_grp = df.groupby("Medu")["G3"].agg(["count", "mean", "median", "std"]).round(2).reset_index()
    medu_grp["education_level"] = medu_grp["Medu"].map({0: "None", 1: "Primary", 2: "5th-9th", 3: "Secondary", 4: "Higher"})
    fedu_grp = df.groupby("Fedu")["G3"].agg(["count", "mean", "median", "std"]).round(2).reset_index()
    fedu_grp["education_level"] = fedu_grp["Fedu"].map({0: "None", 1: "Primary", 2: "5th-9th", 3: "Secondary", 4: "Higher"})
    bivariate_summary["family_education_vs_G3"] = {
        "mother_education": medu_grp,
        "father_education": fedu_grp
    }

    # 5. Internet access vs G3
    internet_grp = df.groupby("internet")["G3"].agg(["count", "mean", "median", "std"]).round(2).reset_index()
    t_internet = stats.ttest_ind(df[df["internet"] == "yes"]["G3"], df[df["internet"] == "no"]["G3"])
    bivariate_summary["internet_vs_G3"] = {
        "grouped": internet_grp,
        "t_stat": round(float(t_internet.statistic), 3),
        "p_value": float(t_internet.pvalue)
    }

    # 6. School support vs G3
    schoolsup_grp = df.groupby("schoolsup")["G3"].agg(["count", "mean", "median", "std"]).round(2).reset_index()
    t_schoolsup = stats.ttest_ind(df[df["schoolsup"] == "yes"]["G3"], df[df["schoolsup"] == "no"]["G3"])
    bivariate_summary["schoolsup_vs_G3"] = {
        "grouped": schoolsup_grp,
        "t_stat": round(float(t_schoolsup.statistic), 3),
        "p_value": float(t_schoolsup.pvalue)
    }

    return bivariate_summary


def compute_correlation_matrix(df, method="pearson"):
    """
    Computes correlation matrix for numerical features.
    """
    num_df = df.select_dtypes(include=[np.number])
    return num_df.corr(method=method).round(3)


if __name__ == "__main__":
    from data_loader import load_dataset
    df = load_dataset()
    uni = compute_univariate_statistics(df)
    print("Univariate Summary (First 5 rows):")
    print(uni.head(5))
    biv = compute_bivariate_analysis(df)
    print("\nStudy time vs G3:\n", biv["studytime_vs_G3"])
    print("\nPrevious grades correlation with G3:\n", biv["previous_grades_vs_G3"])
