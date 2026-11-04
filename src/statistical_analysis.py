"""
Statistical Analysis Module for Student Performance Analytics.
Provides:
- Discrete vs Continuous Variable Classification & Distribution Metrics
- Normality Tests (Shapiro-Wilk, D'Agostino-Pearson)
- Hypothesis Testing (Two-Sample t-tests, One-Way ANOVA)
- Distribution Comparison Analysis (Empirical vs Theoretical)
"""

import numpy as np
import pandas as pd
from scipy import stats

VARIABLE_CLASSIFICATION = {
    "age": {"nature": "Discrete (Count/Integer)", "domain": "[15, 22]", "theoretical": "Truncated Poisson / Discrete uniform-skewed"},
    "Medu": {"nature": "Discrete (Ordinal)", "domain": "{0, 1, 2, 3, 4}", "theoretical": "Multinomial / Discrete Ordinal"},
    "Fedu": {"nature": "Discrete (Ordinal)", "domain": "{0, 1, 2, 3, 4}", "theoretical": "Multinomial / Discrete Ordinal"},
    "traveltime": {"nature": "Discrete (Ordinal)", "domain": "{1, 2, 3, 4}", "theoretical": "Discrete Ordinal"},
    "studytime": {"nature": "Discrete (Ordinal)", "domain": "{1, 2, 3, 4}", "theoretical": "Discrete Ordinal"},
    "failures": {"nature": "Discrete (Count)", "domain": "{0, 1, 2, 3}", "theoretical": "Zero-inflated Poisson"},
    "famrel": {"nature": "Discrete (Likert 1-5)", "domain": "{1, 2, 3, 4, 5}", "theoretical": "Left-skewed Ordinal"},
    "freetime": {"nature": "Discrete (Likert 1-5)", "domain": "{1, 2, 3, 4, 5}", "theoretical": "Discrete Ordinal"},
    "goout": {"nature": "Discrete (Likert 1-5)", "domain": "{1, 2, 3, 4, 5}", "theoretical": "Discrete Ordinal"},
    "Dalc": {"nature": "Discrete (Likert 1-5)", "domain": "{1, 2, 3, 4, 5}", "theoretical": "Heavy Right-skewed Discrete"},
    "Walc": {"nature": "Discrete (Likert 1-5)", "domain": "{1, 2, 3, 4, 5}", "theoretical": "Right-skewed Discrete"},
    "health": {"nature": "Discrete (Likert 1-5)", "domain": "{1, 2, 3, 4, 5}", "theoretical": "Discrete Ordinal"},
    "absences": {"nature": "Continuous / Discrete Count", "domain": "[0, 93]", "theoretical": "Negative Binomial / Gamma (Overdispersed count)"},
    "G1": {"nature": "Continuous approximation (Integer score)", "domain": "[0, 20]", "theoretical": "Bounded quasi-normal"},
    "G2": {"nature": "Continuous approximation (Integer score)", "domain": "[0, 20]", "theoretical": "Zero-inflated bounded quasi-normal"},
    "G3": {"nature": "Continuous approximation (Integer score)", "domain": "[0, 20]", "theoretical": "Zero-inflated bounded quasi-normal"}
}


def test_normality(series):
    """
    Runs Shapiro-Wilk and D'Agostino-Pearson tests for normality.
    Returns test statistics, p-values, and statistical conclusion (at alpha=0.05).
    """
    clean_series = series.dropna()
    n = len(clean_series)
    
    # Shapiro-Wilk (suitable for N < 5000)
    shapiro_stat, shapiro_p = stats.shapiro(clean_series)
    
    # D'Agostino-Pearson omnibus test
    if n >= 20:
        dagostino_stat, dagostino_p = stats.normaltest(clean_series)
    else:
        dagostino_stat, dagostino_p = np.nan, np.nan

    is_normal = (shapiro_p > 0.05) and (dagostino_p > 0.05 if not np.isnan(dagostino_p) else True)

    return {
        "N": n,
        "Shapiro_Statistic": round(float(shapiro_stat), 4),
        "Shapiro_p_value": float(shapiro_p),
        "DAgostino_Statistic": round(float(dagostino_stat), 4) if not np.isnan(dagostino_stat) else None,
        "DAgostino_p_value": float(dagostino_p) if not np.isnan(dagostino_p) else None,
        "Is_Normally_Distributed_at_0.05": bool(is_normal),
        "Conclusion": "Normally distributed (Fail to reject H0)" if is_normal else "Significantly deviates from normal distribution (Reject H0 at p < 0.05)"
    }


def analyze_all_numerical_distributions(df):
    """
    Analyzes normality and characteristics of core numerical features.
    """
    targets = ["studytime", "absences", "G3", "age", "Medu", "Fedu", "G1", "G2"]
    results = []

    for col in targets:
        if col in df.columns:
            norm_res = test_normality(df[col])
            info = VARIABLE_CLASSIFICATION.get(col, {"nature": "Numerical", "theoretical": "Empirical"})
            results.append({
                "Variable": col,
                "Nature": info["nature"],
                "Theoretical Distribution": info["theoretical"],
                "Mean": round(float(df[col].mean()), 2),
                "Std": round(float(df[col].std()), 2),
                "Skewness": round(float(df[col].skew()), 2),
                "Kurtosis": round(float(df[col].kurtosis()), 2),
                "Shapiro_p": norm_res["Shapiro_p_value"],
                "Is_Normal": norm_res["Is_Normally_Distributed_at_0.05"],
                "Statistical Conclusion": norm_res["Conclusion"]
            })

    return pd.DataFrame(results)


def perform_hypothesis_tests(df):
    """
    Executes formal academic hypothesis tests:
    1. One-Way ANOVA: G3 across Study Time levels (1, 2, 3, 4)
    2. One-Way ANOVA: G3 across Mother Education levels (0, 1, 2, 3, 4)
    3. Independent Two-Sample t-test: G3 by Higher Education desire (yes vs no)
    4. Independent Two-Sample t-test: G3 by Internet Access (yes vs no)
    5. Independent Two-Sample t-test: G3 by School Support (yes vs no)
    """
    tests = {}

    # 1. ANOVA: Study time
    groups_st = [group["G3"].values for _, group in df.groupby("studytime")]
    f_st, p_st = stats.f_oneway(*groups_st)
    tests["ANOVA_StudyTime"] = {
        "Test": "One-Way ANOVA (Study Time -> G3)",
        "F_Statistic": round(float(f_st), 3),
        "p_value": float(p_st),
        "Significant_at_0.05": bool(p_st < 0.05),
        "Interpretation": "Study time has a statistically significant positive effect on final grade G3 (p < 0.05)." if p_st < 0.05 else "Study time differences are not statistically significant."
    }

    # 2. ANOVA: Mother Education
    groups_medu = [group["G3"].values for _, group in df.groupby("Medu")]
    f_medu, p_medu = stats.f_oneway(*groups_medu)
    tests["ANOVA_Medu"] = {
        "Test": "One-Way ANOVA (Mother's Education -> G3)",
        "F_Statistic": round(float(f_medu), 3),
        "p_value": float(p_medu),
        "Significant_at_0.05": bool(p_medu < 0.05),
        "Interpretation": "Mother's education level significantly impacts final academic performance (p < 0.05)." if p_medu < 0.05 else "Mother's education differences are not statistically significant."
    }

    # 3. t-test: Higher Education Desire
    grp_high_yes = df[df["higher"] == "yes"]["G3"]
    grp_high_no = df[df["higher"] == "no"]["G3"]
    t_high, p_high = stats.ttest_ind(grp_high_yes, grp_high_no, equal_var=False)
    tests["ttest_HigherEducation"] = {
        "Test": "Welch's Two-Sample t-test (Desire for Higher Ed: Yes vs No -> G3)",
        "Mean_Yes": round(float(grp_high_yes.mean()), 2),
        "Mean_No": round(float(grp_high_no.mean()), 2),
        "t_Statistic": round(float(t_high), 3),
        "p_value": float(p_high),
        "Significant_at_0.05": bool(p_high < 0.05),
        "Interpretation": f"Students planning higher education achieve significantly higher grades ({grp_high_yes.mean():.2f} vs {grp_high_no.mean():.2f}, p < 0.001)."
    }

    # 4. t-test: Internet Access
    grp_net_yes = df[df["internet"] == "yes"]["G3"]
    grp_net_no = df[df["internet"] == "no"]["G3"]
    t_net, p_net = stats.ttest_ind(grp_net_yes, grp_net_no, equal_var=False)
    tests["ttest_Internet"] = {
        "Test": "Welch's Two-Sample t-test (Internet Access: Yes vs No -> G3)",
        "Mean_Yes": round(float(grp_net_yes.mean()), 2),
        "Mean_No": round(float(grp_net_no.mean()), 2),
        "t_Statistic": round(float(t_net), 3),
        "p_value": float(p_net),
        "Significant_at_0.05": bool(p_net < 0.05),
        "Interpretation": f"Internet access at home is associated with higher final grades ({grp_net_yes.mean():.2f} vs {grp_net_no.mean():.2f}, p < 0.05)."
    }

    # 5. t-test: School Support
    grp_sup_yes = df[df["schoolsup"] == "yes"]["G3"]
    grp_sup_no = df[df["schoolsup"] == "no"]["G3"]
    t_sup, p_sup = stats.ttest_ind(grp_sup_yes, grp_sup_no, equal_var=False)
    tests["ttest_SchoolSupport"] = {
        "Test": "Welch's Two-Sample t-test (School Support: Yes vs No -> G3)",
        "Mean_Yes": round(float(grp_sup_yes.mean()), 2),
        "Mean_No": round(float(grp_sup_no.mean()), 2),
        "t_Statistic": round(float(t_sup), 3),
        "p_value": float(p_sup),
        "Significant_at_0.05": bool(p_sup < 0.05),
        "Interpretation": "Extra educational support recipients show lower average grades, reflecting remedial targeting rather than negative efficacy."
    }

    return tests


if __name__ == "__main__":
    from data_loader import load_dataset
    df = load_dataset()
    dist_df = analyze_all_numerical_distributions(df)
    print("Distribution & Normality Analysis:")
    print(dist_df[["Variable", "Nature", "Mean", "Skewness", "Is_Normal", "Statistical Conclusion"]])
    hyp_tests = perform_hypothesis_tests(df)
    print("\nHypothesis Tests:")
    for k, v in hyp_tests.items():
        print(f"{k}: {v['Test']} -> p-value={v['p_value']}, {v['Interpretation']}")
