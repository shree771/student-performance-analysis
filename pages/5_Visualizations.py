"""Page 5: Data Visualizations"""
import os
import sys
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from visualizations import VISUALIZATION_METADATA, CHARTS_DIR

st.set_page_config(page_title="Data Visualizations", page_icon="🎨", layout="wide")
st.title("🎨 Eight Required Data Visualizations")
st.caption("Each visualization includes complete academic metadata, observations, and conclusions.")

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

sel = st.selectbox("Select Chart to View:", viz_names)
fn, meta_key = viz_map[sel]
meta = VISUALIZATION_METADATA[meta_key]
fp = os.path.join(CHARTS_DIR, fn)

col1, col2 = st.columns([1.3, 1])
with col1:
    if os.path.exists(fp):
        st.image(fp, caption=meta["title"], use_container_width=True)
with col2:
    st.subheader(meta["title"])
    st.markdown(f"**What It Shows:** {meta['what_shows']}")
    st.markdown(f"**Why Created:** {meta['why_created']}")
    st.markdown(f"**Key Observations:** {meta['observations']}")
    st.markdown(f"**Academic Interpretation:** {meta['interpretation']}")
    st.markdown(f"**Conclusion:** {meta['conclusion']}")
