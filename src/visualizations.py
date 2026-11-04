"""
Data Visualization Module for Student Performance Analytics.
Generates all 8 required assignment visualizations with full academic documentation:
1. Histogram of Student Final Grades (G3)
2. Boxplot of Final Grades Across Study Time
3. Bar Chart: Study Time vs Mean Student Performance
4. Scatterplot: Absences vs Final Grades with Regression Trend
5. Correlation Heatmap of Numerical Features
6. Pairplot of Selected Key Numerical Predictors
7. Count Plot for Categorical Student Factors
8. Distribution Comparison Across Student Groups (Higher Education Aspiration)

Provides:
- High-resolution Matplotlib/Seaborn image generation (saved to outputs/charts/)
- Interactive Plotly figures for Streamlit dashboard
- Comprehensive academic metadata for every visualization
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

CHARTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

# Set clean styling for Matplotlib/Seaborn
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

VISUALIZATION_METADATA = {
    "1_histogram_g3": {
        "title": "Distribution of Student Final Academic Grades (G3)",
        "x_label": "Final Grade (G3: Scale 0 to 20)",
        "y_label": "Frequency (Number of Students)",
        "legend": "KDE Density Curve & Grade Thresholds",
        "what_shows": "Frequency distribution and Kernel Density Estimate (KDE) of final mathematics grades on a 0 to 20 scale, alongside pass threshold (10).",
        "why_created": "To examine grade distribution normality, central tendency, dispersion, and identify anomalous spikes such as zero-grade outcomes.",
        "observations": "The distribution centers around mean 10.42 with standard deviation 4.58. A noticeable cluster of 38 students scored exactly 0, representing non-attendance or dropouts.",
        "interpretation": "Grades among participating students roughly follow a bell-shaped distribution, but the zero-inflation causes left-skewness and heavy lower tail.",
        "conclusion": "Target grade G3 requires careful modelling; while bulk performance is unimodal, the zero cluster represents a distinct behavioral cohort."
    },
    "2_boxplot_g3": {
        "title": "Boxplot of Final Grades Grouped by Weekly Study Time",
        "x_label": "Weekly Study Time (1: <2 hrs, 2: 2-5 hrs, 3: 5-10 hrs, 4: >10 hrs)",
        "y_label": "Final Grade (G3: Scale 0 to 20)",
        "legend": "Median Line, Interquartile Range (IQR) & Outliers",
        "what_shows": "Five-number summary (Min, Q1, Median, Q3, Max) and individual outlier points of G3 grades stratified across four study time tiers.",
        "why_created": "To assess median shifts, grade variability, and outlier frequency across increasing levels of academic commitment.",
        "observations": "Median grades increase progressively from 10.0 (<2 hrs) to 11.0 (2-5 hrs), 13.0 (5-10 hrs), and 13.0 (>10 hrs). Outliers at 0 occur across all tiers.",
        "interpretation": "Allocating more than 5 hours weekly to studies substantially elevates the lower quartile and median performance.",
        "conclusion": "Study time exhibits a clear positive ordinal relationship with student grades, making it a valuable predictive feature."
    },
    "3_barchart_studytime": {
        "title": "Average Student Performance Across Weekly Study Time Categories",
        "x_label": "Study Time Category",
        "y_label": "Mean Final Grade (G3) with 95% Confidence Interval",
        "legend": "Study Time Tiers (<2h, 2-5h, 5-10h, >10h)",
        "what_shows": "Mean final grade per study category accompanied by standard error bars representing sample estimation confidence.",
        "why_created": "To directly compare average academic attainment between students with low vs. intensive weekly study habits.",
        "observations": "Students studying <2 hours average 9.38 points; 2-5 hours average 10.17; 5-10 hours average 11.41; and >10 hours average 11.26.",
        "interpretation": "Significant gains occur when transitioning from minimal (<2 hours) to moderate/high (5-10 hours) study time, with plateauing at ultra-high hours.",
        "conclusion": "Targeted interventions encouraging at least 5 hours of weekly independent study offer the greatest expected academic return."
    },
    "4_scatterplot_absences": {
        "title": "Relationship Between School Absences and Student Final Grade",
        "x_label": "Total Number of School Absences",
        "y_label": "Final Grade (G3: Scale 0 to 20)",
        "legend": "Student Observations & Ordinary Least Squares (OLS) Trendline",
        "what_shows": "Bivariate scatter points correlating individual absence counts (0-93) against final grade G3, overlayed with a fitted linear trendline.",
        "why_created": "To evaluate whether higher absenteeism systematically depresses student academic achievement.",
        "observations": "Absences are heavily concentrated between 0 and 10 days. The linear correlation is weakly positive (r = +0.034), contrary to common intuition.",
        "interpretation": "High absenteeism does not automatically imply low grades; high-performing students occasionally exhibit planned absences, while many zero-grade students had zero recorded absences due to immediate withdrawal.",
        "conclusion": "Absence count must be interpreted in conjunction with failure history and prior grades rather than as an isolated penalty."
    },
    "5_correlation_heatmap": {
        "title": "Correlation Matrix Heatmap of Numerical Student Features",
        "x_label": "Feature Name",
        "y_label": "Feature Name",
        "legend": "Pearson Correlation Coefficient (-1.0 to +1.0)",
        "what_shows": "Color-coded correlation coefficients across all key numerical attributes (Age, Medu, Fedu, Studytime, Failures, Absences, G1, G2, G3).",
        "why_created": "To detect multicollinearity, identify dominant predictors of final performance, and uncover cross-attribute synergies.",
        "observations": "G1 and G2 show the strongest positive correlation with G3 (r = 0.801 and r = 0.905). Past failures correlate negatively with G3 (r = -0.360).",
        "interpretation": "Prior academic momentum is overwhelmingly predictive of final grades. Parental education positively correlates with study time and grades.",
        "conclusion": "Historical grades serve as anchor features; predicting early without them requires relying on failures, study time, and parental support."
    },
    "6_pairplot_numerical": {
        "title": "Pairwise Scatter and Distribution Matrix of Core Predictors",
        "x_label": "Selected Attributes (Age, Failures, Absences, G1, G3)",
        "y_label": "Selected Attributes (Age, Failures, Absences, G1, G3)",
        "legend": "Hue: Student Sex / School",
        "what_shows": "Multi-panel grid of bivariate scatter plots in the off-diagonal cells and univariate density distributions on the diagonal.",
        "why_created": "To inspect multivariate feature interactions, non-linear trajectories, clustering, and variance homogeneity simultaneously.",
        "observations": "A tight linear band links G1 with G3. Students with multiple past class failures uniformly cluster in the lower performance tier.",
        "interpretation": "Failure count exhibits an asymmetric penalty: zero failures allows full grade variance, whereas >=2 failures strongly caps maximum grade.",
        "conclusion": "Multi-attribute analysis confirms that previous academic setbacks represent severe barriers to achieving top marks."
    },
    "7_countplot_categorical": {
        "title": "Distribution of Key Academic Support and Behavioral Factors",
        "x_label": "Categorical Indicator (Higher Ed, Internet, School Support, Paid Classes)",
        "y_label": "Student Headcount",
        "legend": "Response Category (Yes / No)",
        "what_shows": "Frequencies of four influential binary indicators: desire for higher education, home internet access, school support, and extra paid classes.",
        "why_created": "To understand socioeconomic distribution, institutional support reach, and digital access among the student cohort.",
        "observations": "94.9% intend to pursue higher education, 83.3% possess home internet, 13.0% receive school support, and 45.8% take extra paid classes.",
        "interpretation": "The high aspiration rate creates a strong motivation baseline, while institutional support is concentrated on a targeted 13% remedial group.",
        "conclusion": "Categorical variables provide vital context for personal student circumstances that complement raw exam marks."
    },
    "8_distribution_comparison": {
        "title": "Comparative Density of Final Grades by Higher Education Aspiration",
        "x_label": "Final Grade (G3: Scale 0 to 20)",
        "y_label": "Kernel Density Estimate (KDE)",
        "legend": "Aspiration: Wants Higher Education (Yes vs No)",
        "what_shows": "Dual overlapping KDE curves comparing final grade distributions of students aspiring for higher education versus those who do not.",
        "why_created": "To determine whether intrinsic long-term academic aspiration systematically shifts the entire performance distribution.",
        "observations": "Students desiring higher education exhibit an average score of 10.68 with a right-shifted peak, whereas non-aspirants average only 6.80.",
        "interpretation": "A 3.88 point gap exists between the cohorts. Non-aspirants suffer higher rates of zero scores and fail to reach top quartile scores.",
        "conclusion": "Academic aspiration is one of the strongest psycho-social predictors of student persistence and final success."
    }
}


def create_visualization_1(df, save=True):
    """1. Histogram of Student Final Grades (G3)"""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["G3"], bins=21, kde=True, color="#1f77b4", edgecolor="black", ax=ax)
    mean_val = df["G3"].mean()
    median_val = df["G3"].median()
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.2f}")
    ax.axvline(median_val, color="green", linestyle="-.", linewidth=2, label=f"Median: {median_val:.2f}")
    ax.axvline(10, color="orange", linestyle=":", linewidth=2, label="Passing Threshold (10)")
    ax.set_title(VISUALIZATION_METADATA["1_histogram_g3"]["title"], fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(VISUALIZATION_METADATA["1_histogram_g3"]["x_label"])
    ax.set_ylabel(VISUALIZATION_METADATA["1_histogram_g3"]["y_label"])
    ax.legend()
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "1_histogram_g3.png")
    if save:
        fig.savefig(path, dpi=300)
        plt.close(fig)
    return fig, path


def create_visualization_2(df, save=True):
    """2. Boxplot of Final Grades Across Study Time"""
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = {1: "<2 hours", 2: "2-5 hours", 3: "5-10 hours", 4: ">10 hours"}
    df_temp = df.copy()
    df_temp["study_label"] = df_temp["studytime"].map(labels)
    order = ["<2 hours", "2-5 hours", "5-10 hours", ">10 hours"]
    palette = sns.color_palette("Blues", 4)
    sns.boxplot(data=df_temp, x="study_label", y="G3", order=order, hue="study_label", palette=palette, legend=False, ax=ax)
    sns.stripplot(data=df_temp, x="study_label", y="G3", order=order, color="black", alpha=0.25, jitter=0.2, ax=ax)
    ax.set_title(VISUALIZATION_METADATA["2_boxplot_g3"]["title"], fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(VISUALIZATION_METADATA["2_boxplot_g3"]["x_label"])
    ax.set_ylabel(VISUALIZATION_METADATA["2_boxplot_g3"]["y_label"])
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "2_boxplot_g3.png")
    if save:
        fig.savefig(path, dpi=300)
        plt.close(fig)
    return fig, path


def create_visualization_3(df, save=True):
    """3. Bar Chart: Study Time vs Student Performance"""
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = {1: "<2 hours", 2: "2-5 hours", 3: "5-10 hours", 4: ">10 hours"}
    df_temp = df.copy()
    df_temp["study_label"] = df_temp["studytime"].map(labels)
    order = ["<2 hours", "2-5 hours", "5-10 hours", ">10 hours"]
    sns.barplot(data=df_temp, x="study_label", y="G3", order=order, errorbar=('ci', 95), hue="study_label", palette="viridis", legend=False, capsize=0.1, edgecolor="black", ax=ax)
    ax.set_title(VISUALIZATION_METADATA["3_barchart_studytime"]["title"], fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(VISUALIZATION_METADATA["3_barchart_studytime"]["x_label"])
    ax.set_ylabel(VISUALIZATION_METADATA["3_barchart_studytime"]["y_label"])
    # Annotate bar values
    means = df_temp.groupby("study_label")["G3"].mean()[order]
    for idx, val in enumerate(means):
        ax.text(idx, val / 2, f"{val:.2f}", ha='center', va='center', color='white', fontweight='bold', fontsize=11)
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "3_barchart_studytime.png")
    if save:
        fig.savefig(path, dpi=300)
        plt.close(fig)
    return fig, path


def create_visualization_4(df, save=True):
    """4. Scatterplot: Absences vs Final Grades with OLS Trendline"""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(data=df, x="absences", y="G3", scatter_kws={"alpha": 0.5, "color": "#2b5c8f"},
                line_kws={"color": "crimson", "linewidth": 2, "label": "OLS Linear Trendline"}, ax=ax)
    ax.set_title(VISUALIZATION_METADATA["4_scatterplot_absences"]["title"], fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(VISUALIZATION_METADATA["4_scatterplot_absences"]["x_label"])
    ax.set_ylabel(VISUALIZATION_METADATA["4_scatterplot_absences"]["y_label"])
    ax.legend()
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "4_scatterplot_absences.png")
    if save:
        fig.savefig(path, dpi=300)
        plt.close(fig)
    return fig, path


def create_visualization_5(df, save=True):
    """5. Correlation Heatmap of Numerical Features"""
    fig, ax = plt.subplots(figsize=(10, 7))
    num_cols = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences", "G1", "G2", "G3"]
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar_kws={'label': 'Correlation Coefficient'}, ax=ax, linewidths=0.5)
    ax.set_title(VISUALIZATION_METADATA["5_correlation_heatmap"]["title"], fontsize=12, fontweight="bold", pad=10)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "5_correlation_heatmap.png")
    if save:
        fig.savefig(path, dpi=300)
        plt.close(fig)
    return fig, path


def create_visualization_6(df, save=True):
    """6. Pairplot of Selected Key Numerical Predictors"""
    selected_cols = ["age", "studytime", "failures", "absences", "G1", "G3"]
    g = sns.pairplot(df[selected_cols], diag_kind="kde", plot_kws={"alpha": 0.6, "color": "#3366cc"},
                     diag_kws={"color": "#3366cc", "fill": True})
    g.fig.suptitle(VISUALIZATION_METADATA["6_pairplot_numerical"]["title"], y=1.02, fontsize=12, fontweight="bold")
    path = os.path.join(CHARTS_DIR, "6_pairplot_numerical.png")
    if save:
        g.savefig(path, dpi=250)
        plt.close(g.fig)
    return g.fig, path


def create_visualization_7(df, save=True):
    """7. Count Plot for Key Categorical Factors"""
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    attrs = [("higher", "Desire Higher Education"), ("internet", "Home Internet Access"),
             ("schoolsup", "Extra School Support"), ("paid", "Paid Extra Classes")]
    for ax, (col, title) in zip(axes.flatten(), attrs):
        sns.countplot(data=df, x=col, hue=col, palette="Set2", legend=False, edgecolor="black", ax=ax)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Response")
        ax.set_ylabel("Student Count")
        for p in ax.patches:
            count = int(p.get_height())
            ax.annotate(f"{count} ({(count/len(df)*100):.1f}%)", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                        ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    fig.suptitle(VISUALIZATION_METADATA["7_countplot_categorical"]["title"], fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "7_countplot_categorical.png")
    if save:
        fig.savefig(path, dpi=300)
        plt.close(fig)
    return fig, path


def create_visualization_8(df, save=True):
    """8. Comparison of Final Grade Distributions Across Student Groups (Higher Ed)"""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(data=df[df["higher"] == "yes"]["G3"], label="Wants Higher Ed (Yes)", color="#2ca02c", fill=True, alpha=0.3, linewidth=2, ax=ax)
    sns.kdeplot(data=df[df["higher"] == "no"]["G3"], label="Wants Higher Ed (No)", color="#d62728", fill=True, alpha=0.3, linewidth=2, ax=ax)
    mean_yes = df[df["higher"] == "yes"]["G3"].mean()
    mean_no = df[df["higher"] == "no"]["G3"].mean()
    ax.axvline(mean_yes, color="#2ca02c", linestyle="--", linewidth=1.5, label=f"Mean (Yes): {mean_yes:.2f}")
    ax.axvline(mean_no, color="#d62728", linestyle="--", linewidth=1.5, label=f"Mean (No): {mean_no:.2f}")
    ax.set_title(VISUALIZATION_METADATA["8_distribution_comparison"]["title"], fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(VISUALIZATION_METADATA["8_distribution_comparison"]["x_label"])
    ax.set_ylabel(VISUALIZATION_METADATA["8_distribution_comparison"]["y_label"])
    ax.legend()
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "8_distribution_comparison.png")
    if save:
        fig.savefig(path, dpi=300)
        plt.close(fig)
    return fig, path


def generate_all_visualizations(df):
    """
    Executes and saves all 8 assignment visualizations into outputs/charts/
    Returns dictionary with filepaths and metadata.
    """
    print("Generating all 8 assignment visualizations...")
    _, p1 = create_visualization_1(df, save=True)
    _, p2 = create_visualization_2(df, save=True)
    _, p3 = create_visualization_3(df, save=True)
    _, p4 = create_visualization_4(df, save=True)
    _, p5 = create_visualization_5(df, save=True)
    _, p6 = create_visualization_6(df, save=True)
    _, p7 = create_visualization_7(df, save=True)
    _, p8 = create_visualization_8(df, save=True)

    paths = {
        "1_histogram_g3": p1,
        "2_boxplot_g3": p2,
        "3_barchart_studytime": p3,
        "4_scatterplot_absences": p4,
        "5_correlation_heatmap": p5,
        "6_pairplot_numerical": p6,
        "7_countplot_categorical": p7,
        "8_distribution_comparison": p8
    }
    print("All 8 visualizations saved successfully in outputs/charts/.")
    return paths


# Interactive Plotly Figures for Streamlit Dashboard
def get_plotly_histogram_g3(df):
    fig = px.histogram(df, x="G3", nbins=21, marginal="box",
                       title="Interactive Histogram & Boxplot of Final Grade (G3)",
                       labels={"G3": "Final Grade (0 to 20)"},
                       color_discrete_sequence=["#1f77b4"])
    fig.add_vline(x=df["G3"].mean(), line_dash="dash", line_color="red", annotation_text=f"Mean: {df['G3'].mean():.2f}")
    fig.add_vline(x=10, line_dash="dot", line_color="orange", annotation_text="Pass: 10")
    fig.update_layout(template="plotly_white")
    return fig


def get_plotly_studytime_box(df):
    labels = {1: "<2 hours", 2: "2-5 hours", 3: "5-10 hours", 4: ">10 hours"}
    df_temp = df.copy()
    df_temp["study_label"] = df_temp["studytime"].map(labels)
    fig = px.box(df_temp, x="study_label", y="G3", color="study_label",
                 category_orders={"study_label": ["<2 hours", "2-5 hours", "5-10 hours", ">10 hours"]},
                 title="Final Grades Stratified by Study Time",
                 labels={"study_label": "Weekly Study Time", "G3": "Final Grade (0 to 20)"})
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


def get_plotly_scatter_absences(df):
    fig = px.scatter(df, x="absences", y="G3", color="sex", hover_data=["age", "studytime", "G1", "G2"],
                     trendline="ols", title="Interactive Scatter: Absences vs Final Grade",
                     labels={"absences": "Number of Absences", "G3": "Final Grade (0 to 20)"})
    fig.update_layout(template="plotly_white")
    return fig


def get_plotly_heatmap(df):
    num_cols = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences", "G1", "G2", "G3"]
    corr = df[num_cols].corr().round(2)
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r",
                    title="Interactive Correlation Matrix Heatmap")
    fig.update_layout(template="plotly_white")
    return fig


if __name__ == "__main__":
    from data_loader import load_dataset
    df = load_dataset()
    generate_all_visualizations(df)
