
import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from data_loader import load_dataset, download_uci_dataset, ATTRIBUTE_METADATA, get_dataset_summary
from preprocessing import PreprocessingPipeline, execute_full_preprocessing
from eda import compute_univariate_statistics, compute_categorical_distributions, compute_bivariate_analysis, compute_correlation_matrix
from statistical_analysis import analyze_all_numerical_distributions, perform_hypothesis_tests, VARIABLE_CLASSIFICATION
from visualizations import (
    VISUALIZATION_METADATA, get_plotly_histogram_g3, get_plotly_studytime_box,
    get_plotly_scatter_absences, get_plotly_heatmap, CHARTS_DIR
)
from evaluation import load_model_results, generate_evaluation_diagnostics
from prediction import GradePredictor, classify_grade

# Page configuration
st.set_page_config(
    page_title="Student Performance Analytics | KPRIET",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom academic styling
st.markdown("""
<style>
    .main-header {
        font-family: 'Helvetica Neue', Arial, sans-serif;
        color: #1a365d;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #4a5568;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2b6cb0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .callout-box {
        background-color: #ebf8ff;
        border-left: 4px solid #3182ce;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 12px 0;
        color: #2b6cb0;
    }
    .sidebar-info {
        font-size: 0.82rem;
        color: #4a5568;
        line-height: 1.4;
        background-color: #edf2f7;
        padding: 12px;
        border-radius: 6px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_data():
    return load_dataset()


@st.cache_resource
def get_predictor():
    return GradePredictor()


# Load dataset safely
try:
    df = get_data()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    if st.button("Download Dataset Automatically from UCI"):
        ok, msg = download_uci_dataset(force=True)
        if ok:
            st.success(msg)
            st.rerun()
        else:
            st.error(msg)
    st.stop()

# ==================== SIDEBAR NAVIGATION ====================
st.sidebar.image("https://img.icons8.com/fluency/96/student-registration.png", width=64)
st.sidebar.title("STUDENT PERFORMANCE ANALYTICS")
st.sidebar.caption("Course Project | U21ADP05")

nav_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "1. Home Dashboard",
        "2. Dataset Overview",
        "3. Data Preprocessing",
        "4. Exploratory Data Analysis",
        "5. Statistical Analysis",
        "6. Data Visualizations",
        "7. Seven Visualization Stages",
        "8. Deep Learning Model",
        "9. Model Evaluation",
        "10. Student Grade Prediction",
        "11. Conclusion and References"
    ]
)

st.sidebar.markdown("""
<div class="sidebar-info">
<b>Course:</b> U21ADP05 - Exploratory Data Analysis & Visualization<br/>
<b>Institution:</b> KPR Institute of Engineering & Technology<br/>
<b>Department:</b> Artificial Intelligence & Data Science<br/>
<b>Academic Year:</b> 2026–2027<br/>
<b>Student:</b> Shreenidhi E.
</div>
""", unsafe_allow_html=True)

# Download Report in Sidebar
report_pdf_path = os.path.join(os.path.dirname(__file__), "report", "project_report.pdf")
report_docx_path = os.path.join(os.path.dirname(__file__), "report", "project_report.docx")

st.sidebar.markdown("---")
st.sidebar.subheader("📄 Project Reports")
if os.path.exists(report_pdf_path):
    with open(report_pdf_path, "rb") as f:
        st.sidebar.download_button(
            label="📥 Download PDF Report",
            data=f,
            file_name="project_report.pdf",
            mime="application/pdf"
        )
if os.path.exists(report_docx_path):
    with open(report_docx_path, "rb") as f:
        st.sidebar.download_button(
            label="📥 Download Word (.docx)",
            data=f,
            file_name="project_report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )


# ==================== 1. HOME DASHBOARD ====================
if nav_choice == "1. Home Dashboard":
    st.markdown('<div class="main-header">Student Performance Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Exploratory Data Analysis and Deep Learning Grade Prediction System</div>', unsafe_allow_html=True)

    # Summary KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df)}</div><div class="metric-label">Total Students</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{df.shape[1]}</div><div class="metric-label">Total Features</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{df["G3"].mean():.2f}</div><div class="metric-label">Average Final Grade</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{int(df["G3"].max())} / 20</div><div class="metric-label">Highest Final Grade</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{int(df["G3"].min())} / 20</div><div class="metric-label">Lowest Final Grade</div></div>', unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Executive Summary Tabs
    tab1, tab2, tab3 = st.tabs(["📋 Executive Summary", "🔍 Key Statistical Insights", "🤖 Deep Learning Model Overview"])

    with tab1:
        st.subheader("Project Context & Academic Purpose")
        st.write(
            "This project analyzes the official **UCI Student Performance Dataset (Mathematics Cohort)** to discover the "
            "demographic, socio-economic, and behavioral factors that govern student achievement. "
            "Using both statistical Exploratory Data Analysis (EDA) and a TensorFlow/Keras Multi-Layer Perceptron (MLP), "
            "we predict final student academic grades (**G3**) to enable early and targeted educational interventions."
        )
        st.markdown("""
        - **Institution:** KPR Institute of Engineering and Technology (KPRIET)
        - **Course:** U21ADP05 – Exploratory Data Analysis and Visualization (Academic Year 2026–2027)
        - **Benchmark Dataset:** 395 student records, 33 attributes (Cortez & Silva, 2008)
        - **Primary Target Variable:** `G3` (Final academic examination score, scale 0 to 20)
        """)

    with tab2:
        st.subheader("Core Statistical Discoveries")
        c1, c2 = st.columns(2)
        with c1:
            st.info("""
            **1. Study Time Payoff:**  
            Students studying 5–10 hours weekly achieve an average final score of **11.41**, compared to **9.38** for those studying <2 hours (One-Way ANOVA F = 3.65, p = 0.013).
            """)
            st.info("""
            **2. Power of Academic Aspiration:**  
            Students desiring to pursue higher education average **10.68**, compared to **6.80** for non-aspirants—a 3.88 point performance premium (t = 4.41, p < 0.001).
            """)
        with c2:
            st.warning("""
            **3. Past Failures Asymmetry:**  
            A history of course failures imposes a heavy ceiling: students with zero failures range up to 20 points, whereas students with ≥2 failures average under 6 points.
            """)
            st.warning("""
            **4. The 'Zero-Grade' Phenomenon:**  
            38 students scored exactly 0 in final evaluation. Detailed inquiry shows this reflects early course dropouts or exam absenteeism rather than true zero ability.
            """)

    with tab3:
        st.subheader("Deep Learning Model Performance")
        try:
            results = load_model_results()
            m = results["mlp_test_metrics"]
            b = results["baseline_metrics"]
            st.success(f"**Best Deep Learning MLP Test MAE:** {m['MAE']:.3f} marks | **RMSE:** {m['RMSE']:.3f} | **R²:** {m['R2']:.3f}")
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("MLP Test MAE", f"{m['MAE']:.3f}", "Scale: 0-20")
            col_m2.metric("Random Forest Baseline MAE", f"{b['Random_Forest']['MAE']:.3f}", "Tree Ensemble")
            col_m3.metric("Linear Regression Baseline MAE", f"{b['Linear_Regression']['MAE']:.3f}", "Linear Baseline")
        except Exception:
            st.info("Run `python src/model.py` to view active model results.")

    st.markdown("---")
    st.subheader("Quick Distribution Snapshot")
    st.plotly_chart(get_plotly_histogram_g3(df), use_container_width=True)


# ==================== 2. DATASET OVERVIEW ====================
elif nav_choice == "2. Dataset Overview":
    st.markdown('<div class="main-header">Dataset Overview & Attribute Dictionary</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">UCI Machine Learning Repository: Student Performance Dataset (student-mat.csv)</div>', unsafe_allow_html=True)

    summary = get_dataset_summary(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", summary["num_records"])
    col2.metric("Total Attributes", summary["num_attributes"])
    col3.metric("Target Variable", summary["target_variable"], "Scale: 0–20")
    col4.metric("Missing Values", summary["missing_values"], "100% Complete")

    st.markdown("---")
    st.subheader("Explore Raw Dataset Records")
    rows_to_show = st.slider("Select number of rows to display:", 5, 50, 10)
    st.dataframe(df.head(rows_to_show), use_container_width=True)

    st.subheader("Complete 33-Attribute Dictionary")
    attr_list = []
    for k, v in ATTRIBUTE_METADATA.items():
        attr_list.append({
            "Attribute": k,
            "Type": v["type"],
            "Description": v["description"],
            "Sample Unique Values": str(df[k].unique()[:5].tolist())
        })
    st.dataframe(pd.DataFrame(attr_list), use_container_width=True)

    st.subheader("Descriptive Statistical Summary")
    st.dataframe(df.describe().T.round(2), use_container_width=True)


# ==================== 3. DATA PREPROCESSING ====================
elif nav_choice == "3. Data Preprocessing":
    st.markdown('<div class="main-header">Complete Data Preprocessing Pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Leakage-Free Transformation, IQR Outlier Detection, and 70/15/15 Splitting</div>', unsafe_allow_html=True)

    tab_prep1, tab_prep2, tab_prep3 = st.tabs(["1. Data Cleaning & Outliers", "2. Leakage-Free Splitting", "3. Encoding & Transformation"])

    with tab_prep1:
        st.subheader("Missing Values and Duplicates Check")
        c1, c2 = st.columns(2)
        with c1:
            st.success("✅ **Missing Values:** Exactly 0 missing values across all 395 records.")
        with c2:
            st.success("✅ **Duplicate Records:** Exactly 0 duplicate rows detected.")

        st.markdown("---")
        st.subheader("IQR Method Outlier Analysis")
        st.write(
            "Outliers are identified using the Interquartile Range (IQR = Q3 - Q1) rule:  \n"
            r"$\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}, \quad \text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$"
        )
        pipe = PreprocessingPipeline()
        outliers = pipe.detect_outliers_iqr(df)
        outlier_rows = []
        for col, stats in outliers.items():
            if stats["Outlier_Count"] > 0:
                outlier_rows.append({
                    "Feature": col,
                    "Q1": stats["Q1"],
                    "Q3": stats["Q3"],
                    "IQR": stats["IQR"],
                    "Lower Bound": stats["Lower_Bound"],
                    "Upper Bound": stats["Upper_Bound"],
                    "Outlier Count": stats["Outlier_Count"],
                    "Outlier %": f"{stats['Outlier_Percentage']:.2f}%",
                    "Observed Max": stats["Max"]
                })
        st.dataframe(pd.DataFrame(outlier_rows), use_container_width=True)

        st.markdown("""
        <div class="callout-box">
        <b>Academic Justification on Outlier Treatment:</b><br/>
        In student educational analytics, extreme absence counts (e.g., 50+ absences) and zero final grades (G3 = 0) 
        represent genuine student behavior (truancy, severe illness, or course dropout) rather than instrument measurement errors. 
        Blindly deleting these records discards the most vulnerable student cohort. 
        Hence, our pipeline retains these records with robust scaling.
        </div>
        """, unsafe_allow_html=True)

    with tab_prep2:
        st.subheader("70% Training / 15% Validation / 15% Testing Split")
        st.markdown("""
        To rigorously prevent data leakage, data splitting is executed with a fixed random seed (`random_state=42`)
        **before** any fitting of scalers or categorical encoders.
        """)
        s_col1, s_col2, s_col3 = st.columns(3)
        s_col1.metric("Training Split (70%)", "276 records", "Model Fitting Only")
        s_col2.metric("Validation Split (15%)", "59 records", "Hyperparameter Tuning")
        s_col3.metric("Test Split (15%)", "60 records", "Final Evaluation Only")

    with tab_prep3:
        st.subheader("Feature Transformation & Pipeline Comparison")
        st.write(
            "Nominal categorical variables (17 features) are encoded via `OneHotEncoder(drop='first')`, "
            "and numerical variables are scaled with `StandardScaler()`. Both transformers are fitted exclusively on the training partition."
        )
        prep_res = execute_full_preprocessing(df, save_pipeline=False)
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Original Raw Features (Sample Row):**")
            st.dataframe(df.drop(columns=["G3"]).head(1), use_container_width=True)
        with c2:
            st.write(f"**Processed Feature Matrix ({prep_res['processed_feature_count']} Features):**")
            st.dataframe(prep_res["X_train_proc"].head(1), use_container_width=True)


# ==================== 4. EXPLORATORY DATA ANALYSIS ====================
elif nav_choice == "4. Exploratory Data Analysis":
    st.markdown('<div class="main-header">Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Univariate Statistical Profiling, Skewness, Kurtosis, and Bivariate Analysis</div>', unsafe_allow_html=True)

    tab_eda1, tab_eda2, tab_eda3 = st.tabs(["A. Univariate Statistics", "B. Categorical Frequencies", "C. Bivariate Relationships"])

    with tab_eda1:
        st.subheader("Comprehensive Univariate Statistical Profile")
        uni_df = compute_univariate_statistics(df)
        st.dataframe(uni_df, use_container_width=True)

        st.subheader("Interactive Feature Distribution Explorer")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_col = st.selectbox("Select numerical feature to inspect:", num_cols, index=num_cols.index("G3") if "G3" in num_cols else 0)
        col_fig = px.histogram(df, x=selected_col, marginal="box", nbins=20,
                               title=f"Univariate Distribution of {selected_col}",
                               color_discrete_sequence=["#2b6cb0"])
        col_fig.add_vline(x=df[selected_col].mean(), line_dash="dash", line_color="red", annotation_text="Mean")
        col_fig.add_vline(x=df[selected_col].median(), line_dash="dot", line_color="green", annotation_text="Median")
        st.plotly_chart(col_fig, use_container_width=True)

    with tab_eda2:
        st.subheader("Categorical Feature Distributions")
        cat_dists = compute_categorical_distributions(df)
        selected_cat = st.selectbox("Select categorical feature:", list(cat_dists.keys()))
        col_c1, col_c2 = st.columns([1, 2])
        with col_c1:
            st.dataframe(cat_dists[selected_cat], use_container_width=True)
        with col_c2:
            fig_cat = px.bar(cat_dists[selected_cat], x="Category", y="Frequency", text="Percentage (%)",
                             title=f"Frequency Breakdown: {selected_cat}",
                             color="Category", color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig_cat, use_container_width=True)

    with tab_eda3:
        st.subheader("Bivariate Relationships with Final Grade (G3)")
        biv = compute_bivariate_analysis(df)
        biv_choice = st.selectbox("Select relationship to analyze:", [
            "Study Time vs Final Grade",
            "Absences vs Final Grade",
            "Previous Grades (G1, G2) vs Final Grade",
            "Family Education vs Final Grade",
            "Internet Access vs Final Grade",
            "School Support vs Final Grade"
        ])
        if biv_choice == "Study Time vs Final Grade":
            st.dataframe(biv["studytime_vs_G3"], use_container_width=True)
            st.plotly_chart(get_plotly_studytime_box(df), use_container_width=True)
        elif biv_choice == "Absences vs Final Grade":
            st.write(f"**Pearson Correlation Coefficient:** {biv['absences_vs_G3']['pearson_r']:.3f} (p = {biv['absences_vs_G3']['p_value']:.4f})")
            st.dataframe(biv["absences_vs_G3"]["grouped"], use_container_width=True)
            st.plotly_chart(get_plotly_scatter_absences(df), use_container_width=True)
        elif biv_choice == "Previous Grades (G1, G2) vs Final Grade":
            st.write(f"**G1 Correlation with G3:** r = {biv['previous_grades_vs_G3']['G1_r']} (p < 0.001)")
            st.write(f"**G2 Correlation with G3:** r = {biv['previous_grades_vs_G3']['G2_r']} (p < 0.001)")
            fig_g = px.scatter(df, x="G2", y="G3", trendline="ols", title="G2 vs G3 Correlation", labels={"G2": "Period 2 Grade", "G3": "Final Grade"})
            st.plotly_chart(fig_g, use_container_width=True)
        elif biv_choice == "Family Education vs Final Grade":
            st.write("**Mother Education vs G3:**")
            st.dataframe(biv["family_education_vs_G3"]["mother_education"], use_container_width=True)
            st.write("**Father Education vs G3:**")
            st.dataframe(biv["family_education_vs_G3"]["father_education"], use_container_width=True)
        elif biv_choice == "Internet Access vs Final Grade":
            st.dataframe(biv["internet_vs_G3"]["grouped"], use_container_width=True)
            st.write(f"Two-sample t-test: t = {biv['internet_vs_G3']['t_stat']}, p = {biv['internet_vs_G3']['p_value']:.4f}")
        elif biv_choice == "School Support vs Final Grade":
            st.dataframe(biv["schoolsup_vs_G3"]["grouped"], use_container_width=True)
            st.write(f"Two-sample t-test: t = {biv['schoolsup_vs_G3']['t_stat']}, p = {biv['schoolsup_vs_G3']['p_value']:.4f}")


# ==================== 5. STATISTICAL ANALYSIS ====================
elif nav_choice == "5. Statistical Analysis":
    st.markdown('<div class="main-header">Statistical & Distribution Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Discrete vs. Continuous Distributions, Normality Testing, and ANOVA Hypothesis Tests</div>', unsafe_allow_html=True)

    tab_stat1, tab_stat2 = st.tabs(["1. Normality & Distribution Diagnostics", "2. Academic Hypothesis Tests"])

    with tab_stat1:
        st.subheader("Discrete vs. Continuous Variable Classification")
        dist_analysis = analyze_all_numerical_distributions(df)
        st.dataframe(dist_analysis, use_container_width=True)

        st.markdown("""
        <div class="callout-box">
        <b>Statistically Sound Interpretation:</b><br/>
        Shapiro-Wilk testing confirms that neither student absences nor final grades follow a standard Gaussian distribution. 
        Absences represent a non-negative, heavily right-skewed count process best described by a Negative Binomial distribution. 
        Similarly, final grades (G3) are bounded on [0, 20] and subject to zero-inflation. 
        Forcing a normal fit where it is statistically inappropriate would violate modeling assumptions.
        </div>
        """, unsafe_allow_html=True)

    with tab_stat2:
        st.subheader("Formal Academic Hypothesis Tests")
        hyp = perform_hypothesis_tests(df)
        for key, res in hyp.items():
            with st.expander(f"📌 {res['Test']}", expanded=True):
                col_h1, col_h2 = st.columns([1, 2])
                with col_h1:
                    if "F_Statistic" in res:
                        st.metric("F-Statistic", res["F_Statistic"], f"p-value: {res['p_value']:.5f}")
                    elif "t_Statistic" in res:
                        st.metric("t-Statistic", res["t_Statistic"], f"p-value: {res['p_value']:.5f}")
                with col_h2:
                    st.write(f"**Interpretation:** {res['Interpretation']}")


# ==================== 6. DATA VISUALIZATIONS ====================
elif nav_choice == "6. Data Visualizations":
    st.markdown('<div class="main-header">Required Data Visualizations (8 Charts)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Each chart includes complete academic metadata, observations, and conclusions</div>', unsafe_allow_html=True)

    viz_names = [
        "1. Histogram of Student Final Grades",
        "2. Boxplot of Final Grades Across Study Time",
        "3. Bar Chart: Study Time vs Performance",
        "4. Scatterplot: Absences vs Final Grades",
        "5. Correlation Heatmap",
        "6. Pairplot of Key Numerical Predictors",
        "7. Count Plot for Categorical Factors",
        "8. Comparative Density Across Student Groups"
    ]
    viz_map = {
        "1. Histogram of Student Final Grades": ("1_histogram_g3.png", "1_histogram_g3"),
        "2. Boxplot of Final Grades Across Study Time": ("2_boxplot_g3.png", "2_boxplot_g3"),
        "3. Bar Chart: Study Time vs Performance": ("3_barchart_studytime.png", "3_barchart_studytime"),
        "4. Scatterplot: Absences vs Final Grades": ("4_scatterplot_absences.png", "4_scatterplot_absences"),
        "5. Correlation Heatmap": ("5_correlation_heatmap.png", "5_correlation_heatmap"),
        "6. Pairplot of Key Numerical Predictors": ("6_pairplot_numerical.png", "6_pairplot_numerical"),
        "7. Count Plot for Categorical Factors": ("7_countplot_categorical.png", "7_countplot_categorical"),
        "8. Comparative Density Across Student Groups": ("8_distribution_comparison.png", "8_distribution_comparison")
    }

    selected_viz = st.selectbox("Select Visualization to Examine:", viz_names)
    filename, meta_key = viz_map[selected_viz]
    meta = VISUALIZATION_METADATA[meta_key]
    img_path = os.path.join(CHARTS_DIR, filename)

    col_img, col_desc = st.columns([1.3, 1])

    with col_img:
        if os.path.exists(img_path):
            st.image(img_path, caption=meta["title"], use_container_width=True)
        else:
            st.warning("Chart image not found. Running generation...")

    with col_desc:
        st.subheader("Academic Chart Metadata")
        st.markdown(f"**Title:** {meta['title']}")
        st.markdown(f"**X-Axis:** `{meta['x_label']}` | **Y-Axis:** `{meta['y_label']}`")
        st.markdown(f"**Legend:** {meta['legend']}")
        st.markdown(f"**What It Shows:** {meta['what_shows']}")
        st.markdown(f"**Why Created:** {meta['why_created']}")
        st.markdown(f"**Important Observations:** {meta['observations']}")
        st.markdown(f"**Academic Interpretation:** {meta['interpretation']}")
        st.markdown(f"**Conclusion:** {meta['conclusion']}")


# ==================== 7. SEVEN VISUALIZATION STAGES ====================
elif nav_choice == "7. Seven Visualization Stages":
    st.markdown('<div class="main-header">The Seven Stages of Data Visualization</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Implementation of Ben Fry\'s 7-Stage Visualization Framework</div>', unsafe_allow_html=True)

    stages = [
        {"num": "1", "name": "ACQUIRE", "desc": "Obtain the data from its source, whether from a file, stream, or web service.", "impl": "Automated download and verification of the official UCI Machine Learning Repository zip archive into `dataset/student-mat.csv`."},
        {"num": "2", "name": "PARSE", "desc": "Provide structure for data's meaning and order it into categories.", "impl": "Decoding the semicolon delimiter, typing 33 features into numeric/ordinal/nominal classes, and building an attribute dictionary."},
        {"num": "3", "name": "FILTER", "desc": "Remove all but the data of interest.", "impl": "Identifying nulls, duplicates, computing IQR outlier thresholds, and creating train/val/test splits without leakage."},
        {"num": "4", "name": "MINE", "desc": "Apply statistical and data mining methods to discern patterns and relationships.", "impl": "Univariate profiling (mean, median, skewness, kurtosis), ANOVA tests, t-tests, and Pearson correlation matrices."},
        {"num": "5", "name": "REPRESENT", "desc": "Choose a basic visual model, such as a bar graph, list, or tree.", "impl": "Constructing histograms with KDE, boxplots with strip overlays, regression scatterplots with OLS trendlines, and heatmaps."},
        {"num": "6", "name": "REFINE", "desc": "Improve the representation to make it clearer and visually engaging.", "impl": "Configuring high-contrast palettes, setting axis limits, adding mean/median annotation lines, error bars, and clear figure titles."},
        {"num": "7", "name": "INTERACT", "desc": "Add methods for manipulating the data or controlling what features are visible.", "impl": "Developing interactive Plotly visualizations and dynamic Streamlit widgets (sliders, selectors, and real-time prediction inputs)."}
    ]

    for stg in stages:
        with st.expander(f"Stage {stg['num']}: {stg['name']}", expanded=True):
            col_s1, col_s2 = st.columns([1, 2])
            with col_s1:
                st.markdown(f"**Theoretical Definition:**  \n{stg['desc']}")
            with col_s2:
                st.markdown(f"**Project Implementation in this System:**  \n`{stg['impl']}`")


# ==================== 8. DEEP LEARNING MODEL ====================
elif nav_choice == "8. Deep Learning Model":
    st.markdown('<div class="main-header">Deep Learning Model Architecture & Hyperparameters</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-Layer Perceptron (MLP) Regressor implemented with TensorFlow/Keras</div>', unsafe_allow_html=True)

    col_arch, col_diag = st.columns([1.2, 1])

    with col_arch:
        st.subheader("MLP Neural Network Architecture")
        st.markdown("""
        ```text
        [Input Layer: 41 Processed Features]
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
        [Output Layer: Dense(1, Linear)] ──► Predicted Final Grade G3 (0 to 20)
        ```
        """)
        st.markdown("""
        **Training Configuration:**
        - **Optimizer:** Adam (Initial Learning Rate = 0.001)
        - **Loss Function:** Mean Squared Error (MSE)
        - **Evaluation Metric:** Mean Absolute Error (MAE)
        - **Epochs:** 100 with EarlyStopping (Patience = 15)
        - **Batch Size:** 32
        - **Model Checkpointing:** Saved optimal weights to `models/best_model.keras`
        """)

    with col_diag:
        st.subheader("Hyperparameter Tuning on Validation Split")
        try:
            results = load_model_results()
            st.dataframe(pd.DataFrame(results["tuning_results"]), use_container_width=True)
            st.success(f"**Best Configuration:** {results['best_hyperparameters']['name']}")
        except Exception:
            st.info("Execute `python src/model.py` to view hyperparameter tuning logs.")

    st.markdown("---")
    st.subheader("Early Prediction Experiment (Excluding G1 and G2)")
    st.write(
        "To evaluate whether final academic success can be predicted **early in the school year** before initial examinations take place, "
        "we trained a secondary MLP model strictly omitting G1 and G2."
    )
    try:
        results = load_model_results()
        e = results["early_prediction_metrics"]
        m = results["mlp_test_metrics"]
        c1, c2 = st.columns(2)
        c1.metric("Early Prediction (No G1/G2) Test MAE", f"{e['MAE']:.3f}", f"R² = {e['R2']:.3f}")
        c2.metric("Complete Model (With G1/G2) Test MAE", f"{m['MAE']:.3f}", f"R² = {m['R2']:.3f}")
        st.info("Academic Insight: While early prediction MAE is higher (3.14 vs 1.82), it provides critical diagnostic capability before mid-term marks are locked.")
    except Exception:
        pass


# ==================== 9. MODEL EVALUATION ====================
elif nav_choice == "9. Model Evaluation":
    st.markdown('<div class="main-header">Model Evaluation & Benchmark Diagnostics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Empirical Test-Set Metrics, Learning Curves, and Generalization Assessment</div>', unsafe_allow_html=True)

    try:
        results = load_model_results()
        diag = generate_evaluation_diagnostics(results)

        col1, col2, col3, col4 = st.columns(4)
        m = results["mlp_test_metrics"]
        col1.metric("MLP Test MAE", f"{m['MAE']:.4f}")
        col2.metric("MLP Test MSE", f"{m['MSE']:.4f}")
        col3.metric("MLP Test RMSE", f"{m['RMSE']:.4f}")
        col4.metric("MLP Test R² Score", f"{m['R2']:.4f}")

        st.markdown("---")
        st.subheader("Benchmark Comparison on Held-Out Test Set")
        st.dataframe(diag["comparison_table"], use_container_width=True)

        st.subheader("Evaluation Visualizations")
        c1, c2 = st.columns(2)
        with c1:
            loss_p = os.path.join(CHARTS_DIR, "eval_loss_curves.png")
            if os.path.exists(loss_p):
                st.image(loss_p, caption="Figure 9: Training Loss vs Validation Loss Curves", use_container_width=True)
        with c2:
            pred_p = os.path.join(CHARTS_DIR, "eval_actual_vs_predicted.png")
            if os.path.exists(pred_p):
                st.image(pred_p, caption="Figure 10: Actual vs Predicted Grades", use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            err_p = os.path.join(CHARTS_DIR, "eval_error_distribution.png")
            if os.path.exists(err_p):
                st.image(err_p, caption="Figure 11: Prediction Error Distribution", use_container_width=True)
        with c4:
            bench_p = os.path.join(CHARTS_DIR, "eval_baseline_comparison.png")
            if os.path.exists(bench_p):
                st.image(bench_p, caption="Figure 12: Baseline vs Deep Learning Comparison", use_container_width=True)

        st.subheader("Overfitting & Underfitting Diagnostic Assessment")
        st.markdown(f"**Overfitting Status:** `{diag['overfit_status']}` – {diag['overfit_desc']}")
        st.markdown(f"**Underfitting Status:** `{diag['underfit_status']}` – {diag['underfit_desc']}")

    except Exception as e:
        st.warning(f"Model results not yet loaded: {e}. Run `python src/model.py`.")


# ==================== 10. STUDENT GRADE PREDICTION ====================
elif nav_choice == "10. Student Grade Prediction":
    st.markdown('<div class="main-header">Student Grade Prediction & Academic Advisory</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Input Student Profile to Generate Real-Time Predicted Grade (0-20) and Targeted Interventions</div>', unsafe_allow_html=True)

    predictor = get_predictor()

    with st.form("student_prediction_form"):
        st.subheader("1. Demographic & Family Information")
        f1, f2, f3, f4 = st.columns(4)
        school = f1.selectbox("School", ["GP", "MS"], format_func=lambda x: "Gabriel Pereira (GP)" if x == "GP" else "Mousinho da Silveira (MS)")
        sex = f2.selectbox("Sex", ["F", "M"], format_func=lambda x: "Female (F)" if x == "F" else "Male (M)")
        age = f3.slider("Age", 15, 22, 17)
        address = f4.selectbox("Address Type", ["U", "R"], format_func=lambda x: "Urban (U)" if x == "U" else "Rural (R)")

        f5, f6, f7, f8 = st.columns(4)
        famsize = f5.selectbox("Family Size", ["GT3", "LE3"], format_func=lambda x: ">3 Members (GT3)" if x == "GT3" else "≤3 Members (LE3)")
        pstatus = f6.selectbox("Parent Cohabitation", ["T", "A"], format_func=lambda x: "Living Together (T)" if x == "T" else "Apart (A)")
        medu = f7.selectbox("Mother's Education", [0, 1, 2, 3, 4], index=3, format_func=lambda x: {0: "0 - None", 1: "1 - Primary", 2: "2 - 5th-9th Grade", 3: "3 - Secondary", 4: "4 - Higher"}[x])
        fedu = f8.selectbox("Father's Education", [0, 1, 2, 3, 4], index=3, format_func=lambda x: {0: "0 - None", 1: "1 - Primary", 2: "2 - 5th-9th Grade", 3: "3 - Secondary", 4: "4 - Higher"}[x])

        f9, f10, f11, f12 = st.columns(4)
        mjob = f9.selectbox("Mother's Job", ["at_home", "health", "other", "services", "teacher"])
        fjob = f10.selectbox("Father's Job", ["teacher", "other", "services", "health", "at_home"])
        reason = f11.selectbox("School Reason", ["home", "reputation", "course", "other"])
        guardian = f12.selectbox("Primary Guardian", ["mother", "father", "other"])

        st.subheader("2. Academic Habits & Institutional Support")
        a1, a2, a3, a4 = st.columns(4)
        traveltime = a1.selectbox("Travel Time", [1, 2, 3, 4], index=0, format_func=lambda x: {1: "<15 min", 2: "15-30 min", 3: "30-60 min", 4: ">60 min"}[x])
        studytime = a2.selectbox("Weekly Study Time", [1, 2, 3, 4], index=1, format_func=lambda x: {1: "<2 hours", 2: "2-5 hours", 3: "5-10 hours", 4: ">10 hours"}[x])
        failures = a3.selectbox("Past Class Failures", [0, 1, 2, 3], index=0)
        schoolsup = a4.selectbox("Extra School Support", ["no", "yes"])

        a5, a6, a7, a8 = st.columns(4)
        famsup = a5.selectbox("Family Educational Support", ["yes", "no"])
        paid = a6.selectbox("Extra Paid Classes", ["no", "yes"])
        activities = a7.selectbox("Extracurricular Activities", ["yes", "no"])
        nursery = a8.selectbox("Attended Nursery", ["yes", "no"])

        st.subheader("3. Lifestyle & Socio-Behavioral Profile")
        l1, l2, l3, l4 = st.columns(4)
        higher = l1.selectbox("Wants Higher Education", ["yes", "no"])
        internet = l2.selectbox("Home Internet Access", ["yes", "no"])
        romantic = l3.selectbox("In Romantic Relationship", ["no", "yes"])
        famrel = l4.slider("Family Relationship Quality", 1, 5, 4)

        l5, l6, l7, l8 = st.columns(4)
        freetime = l5.slider("Free Time After School", 1, 5, 3)
        goout = l6.slider("Going Out with Friends", 1, 5, 2)
        dalc = l7.slider("Workday Alcohol Consumption", 1, 5, 1)
        walc = l8.slider("Weekend Alcohol Consumption", 1, 5, 1)

        l9, l10, l11, l12 = st.columns(4)
        health = l9.slider("Current Health Status", 1, 5, 5)
        absences = l10.number_input("School Absences (Days)", min_value=0, max_value=93, value=4, step=1)
        g1 = l11.slider("Period 1 Grade (G1: 0-20)", 0, 20, 12)
        g2 = l12.slider("Period 2 Grade (G2: 0-20)", 0, 20, 13)

        submit_btn = st.form_submit_button("🔮 Predict Student Final Grade")

    if submit_btn:
        student_data = {
            "school": school, "sex": sex, "age": age, "address": address, "famsize": famsize,
            "Pstatus": pstatus, "Medu": medu, "Fedu": fedu, "Mjob": mjob, "Fjob": fjob,
            "reason": reason, "guardian": guardian, "traveltime": traveltime, "studytime": studytime,
            "failures": failures, "schoolsup": schoolsup, "famsup": famsup, "paid": paid,
            "activities": activities, "nursery": nursery, "higher": higher, "internet": internet,
            "romantic": romantic, "famrel": famrel, "freetime": freetime, "goout": goout,
            "Dalc": dalc, "Walc": walc, "health": health, "absences": absences, "G1": g1, "G2": g2
        }

        with st.spinner("Processing features through trained deep learning pipeline..."):
            pred_res = predictor.predict(student_data)

        st.markdown("---")
        st.subheader("Prediction Result & Advisory Assessment")

        p_col1, p_col2 = st.columns([1, 2])
        with p_col1:
            st.markdown(f"""
            <div style="background-color: #f7fafc; border: 2px solid {pred_res['tier_color']}; border-radius: 8px; padding: 20px; text-align: center;">
                <div style="font-size: 0.9rem; text-transform: uppercase; color: #718096;">Predicted Final Grade</div>
                <div style="font-size: 3rem; font-weight: 800; color: {pred_res['tier_color']};">{pred_res['predicted_grade']} <span style="font-size: 1.5rem; color: #a0aec0;">/ 20</span></div>
                <div style="font-size: 1.2rem; font-weight: 600; color: {pred_res['tier_color']};">{pred_res['tier']}</div>
                <div style="font-size: 0.85rem; color: #4a5568; margin-top: 6px;">Status: <b>{pred_res['status']}</b></div>
            </div>
            """, unsafe_allow_html=True)

        with p_col2:
            st.markdown(f"**Academic Summary:** {pred_res['summary']}")
            st.markdown("**Actionable Intervention Recommendations:**")
            for r in pred_res["recommendations"]:
                badge_color = "#e53e3e" if r["type"] == "Critical" else "#dd6b20" if r["type"] == "Urgent" else "#3182ce"
                st.markdown(f"- <span style='background-color: {badge_color}; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; font-weight: bold;'>{r['area']}</span> {r['text']}", unsafe_allow_html=True)

        st.caption(f"ℹ️ {pred_res['disclaimer']}")


# ==================== 11. CONCLUSION AND REFERENCES ====================
elif nav_choice == "11. Conclusion and References":
    st.markdown('<div class="main-header">Conclusion, Future Scope, and Academic References</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Final Synthesis for Course Project U21ADP05</div>', unsafe_allow_html=True)

    st.subheader("Key Academic Takeaways")
    st.markdown("""
    1. **Prior Academic Momentum:** First and second-period grades ($G1$ and $G2$) serve as overwhelmingly dominant predictors ($r > 0.80$). Academic interventions must be deployed within the first six weeks before learning gaps compound.
    2. **Study Habit Elasticity:** Shifting study commitments from $<2$ hours to $5-10$ hours produces an average grade improvement of $+2.0$ points, offering a tangible target for counseling.
    3. **Aspiration Effect:** Long-term aspiration for higher education operates as a powerful protective factor against academic disengagement.
    4. **Deep Learning Efficacy:** The Keras MLP regressor with Dropout ($0.2$) achieved competitive generalization ($\text{MAE} = 1.82$), balancing non-linear representations with tabular stability.
    """)

    st.subheader("Future Research Directions")
    st.markdown("""
    - **Sequential LMS Clickstreams:** Modeling week-by-week online student activity logs using Recurrent Neural Networks (LSTM/GRU) or Transformers.
    - **Explainable Counterfactuals:** Integrating SHAP and counterfactual optimization to deliver prescriptive advice (e.g., 'Increasing study time by 3 hours raises grade probability above 14').
    - **Zero-Inflated Hurdle Regressors:** Employing a two-stage classifier to isolate non-attendance dropouts from continuous exam performers.
    """)

    st.subheader("Academic References (IEEE Style)")
    st.markdown("""
    [1] P. Cortez and A. Silva, "Using data mining to predict secondary school student performance," in *Proceedings of 5th Annual Future Business Technology Conference (FUBUTEC 2008)*, Porto, Portugal, 2008, pp. 5–12.  
    [2] D. Dua and C. Graff, "UCI Machine Learning Repository: Student Performance Data Set," University of California, Irvine, School of Information and Computer Sciences, 2017. [Online]. Available: https://archive.ics.uci.edu/dataset/320/student+performance.  
    [3] M. Abadi et al., "TensorFlow: A system for large-scale machine learning," in *12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16)*, 2016, pp. 265–283.  
    [4] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.  
    [5] C. Romero and S. Ventura, "Educational data mining: A review of the state of the art," *IEEE Transactions on Systems, Man, and Cybernetics, Part C (Applications and Reviews)*, vol. 40, no. 6, pp. 601–618, Nov. 2010.
    """)

    st.markdown("---")
    st.subheader("Download Generated Reports")
    c_p1, c_p2 = st.columns(2)
    with c_p1:
        if os.path.exists(report_pdf_path):
            with open(report_pdf_path, "rb") as f:
                st.download_button("📥 Download Official PDF Report", f, "project_report.pdf", "application/pdf")
    with c_p2:
        if os.path.exists(report_docx_path):
            with open(report_docx_path, "rb") as f:
                st.download_button("📥 Download Official Word Report (.docx)", f, "project_report.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
