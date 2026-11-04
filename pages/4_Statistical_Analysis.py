"""Page 4: Statistical Analysis"""
import os
import sys
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from data_loader import load_dataset
from statistical_analysis import analyze_all_numerical_distributions, perform_hypothesis_tests

st.set_page_config(page_title="Statistical Analysis", page_icon="🔬", layout="wide")
st.title("🔬 Statistical & Distribution Analysis")

df = load_dataset()

st.subheader("Discrete vs Continuous Distributions & Normality Testing")
dists = analyze_all_numerical_distributions(df)
st.dataframe(dists, use_container_width=True)

st.markdown("---")
st.subheader("Academic Hypothesis Tests (ANOVA & Two-Sample t-tests)")
hyp = perform_hypothesis_tests(df)
for k, v in hyp.items():
    with st.expander(f"📌 {v['Test']}", expanded=True):
        st.write(f"**Interpretation:** {v['Interpretation']}")
        st.write(f"**p-value:** `{v['p_value']:.5e}` (Significant at alpha=0.05: `{v['Significant_at_0.05']}`)")
