"""Page 1: Dataset Overview"""
import os
import sys
import streamlit as st
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from data_loader import load_dataset, ATTRIBUTE_METADATA, get_dataset_summary

st.set_page_config(page_title="Dataset Overview", page_icon="📊", layout="wide")
st.title("📊 Dataset Overview & Attribute Dictionary")
st.caption("UCI Machine Learning Repository: Student Performance Dataset (student-mat.csv)")

df = load_dataset()
summary = get_dataset_summary(df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", summary["num_records"])
col2.metric("Total Attributes", summary["num_attributes"])
col3.metric("Target Variable", summary["target_variable"], "Scale: 0–20")
col4.metric("Missing Values", summary["missing_values"], "Complete")

st.markdown("---")
st.subheader("Explore Raw Dataset Records")
rows = st.slider("Select rows to view:", 5, 50, 10)
st.dataframe(df.head(rows), use_container_width=True)

st.subheader("33-Attribute Dictionary")
attr_list = [{"Attribute": k, "Type": v["type"], "Description": v["description"]} for k, v in ATTRIBUTE_METADATA.items()]
st.dataframe(pd.DataFrame(attr_list), use_container_width=True)

st.subheader("Descriptive Statistical Summary")
st.dataframe(df.describe().T.round(2), use_container_width=True)
