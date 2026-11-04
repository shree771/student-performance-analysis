"""Page 2: Data Preprocessing"""
import os
import sys
import streamlit as st
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from data_loader import load_dataset
from preprocessing import PreprocessingPipeline, execute_full_preprocessing

st.set_page_config(page_title="Data Preprocessing", page_icon="⚙️", layout="wide")
st.title("⚙️ Data Preprocessing Pipeline")
st.caption("Missing Values, Duplicates, IQR Outlier Detection, 70/15/15 Splitting & Leakage Prevention")

df = load_dataset()
pipe = PreprocessingPipeline()

col1, col2 = st.columns(2)
col1.success(f"Missing Values: {df.isnull().sum().sum()} nulls detected across 395 records.")
col2.success(f"Duplicate Rows: {df.duplicated().sum()} duplicates detected.")

st.markdown("---")
st.subheader("IQR Method Outlier Analysis")
outliers = pipe.detect_outliers_iqr(df)
outlier_rows = []
for col, s in outliers.items():
    if s["Outlier_Count"] > 0:
        outlier_rows.append({
            "Feature": col, "Q1": s["Q1"], "Q3": s["Q3"], "IQR": s["IQR"],
            "Lower Bound": s["Lower_Bound"], "Upper Bound": s["Upper_Bound"],
            "Outlier Count": s["Outlier_Count"], "Outlier %": f"{s['Outlier_Percentage']:.2f}%"
        })
st.dataframe(pd.DataFrame(outlier_rows), use_container_width=True)

st.info("Outlier Justification: Extreme absences and zero grades reflect real educational behavioral factors (truancy, withdrawal). They are retained with robust scaling.")

st.markdown("---")
st.subheader("70% Train / 15% Validation / 15% Test Split")
s1, s2, s3 = st.columns(3)
s1.metric("Training Set (70%)", "276 records", "Fitted strictly on train")
s2.metric("Validation Set (15%)", "59 records", "Hyperparameter tuning")
s3.metric("Test Set (15%)", "60 records", "Final evaluation only")

prep_res = execute_full_preprocessing(df, save_pipeline=False)
st.write(f"Processed Feature Dimension after One-Hot Encoding: **{prep_res['processed_feature_count']} features**.")
st.dataframe(prep_res["X_train_proc"].head(3), use_container_width=True)
