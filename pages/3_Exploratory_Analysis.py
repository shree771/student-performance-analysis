"""Page 3: Exploratory Analysis"""
import os
import sys
import streamlit as st
import plotly.express as px
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from data_loader import load_dataset
from eda import compute_univariate_statistics, compute_categorical_distributions, compute_bivariate_analysis

st.set_page_config(page_title="Exploratory Analysis", page_icon="📈", layout="wide")
st.title("📈 Exploratory Data Analysis (EDA)")

df = load_dataset()

st.subheader("Univariate Statistical Profile")
uni_df = compute_univariate_statistics(df)
st.dataframe(uni_df, use_container_width=True)

st.markdown("---")
st.subheader("Interactive Numerical Distribution Viewer")
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feat = st.selectbox("Select feature to plot:", num_cols, index=num_cols.index("G3") if "G3" in num_cols else 0)
fig = px.histogram(df, x=feat, marginal="box", nbins=20, title=f"Distribution of {feat}", color_discrete_sequence=["#1f77b4"])
fig.add_vline(x=df[feat].mean(), line_dash="dash", line_color="red", annotation_text="Mean")
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Bivariate Relationships with Final Grade (G3)")
biv = compute_bivariate_analysis(df)
st.write("**Study Time vs Final Grade:**")
st.dataframe(biv["studytime_vs_G3"], use_container_width=True)
