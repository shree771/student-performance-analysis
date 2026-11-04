"""Page 7: Model Evaluation"""
import os
import sys
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from evaluation import load_model_results, generate_evaluation_diagnostics
from visualizations import CHARTS_DIR

st.set_page_config(page_title="Model Evaluation", page_icon="🎯", layout="wide")
st.title("🎯 Model Evaluation & Benchmark Diagnostics")

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

    c1, c2 = st.columns(2)
    with c1:
        lp = os.path.join(CHARTS_DIR, "eval_loss_curves.png")
        if os.path.exists(lp):
            st.image(lp, caption="Figure 9: Training Loss vs Validation Loss Curves", use_container_width=True)
    with c2:
        pp = os.path.join(CHARTS_DIR, "eval_actual_vs_predicted.png")
        if os.path.exists(pp):
            st.image(pp, caption="Figure 10: Actual vs Predicted Grades", use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        ep = os.path.join(CHARTS_DIR, "eval_error_distribution.png")
        if os.path.exists(ep):
            st.image(ep, caption="Figure 11: Prediction Error Distribution", use_container_width=True)
    with c4:
        bp = os.path.join(CHARTS_DIR, "eval_baseline_comparison.png")
        if os.path.exists(bp):
            st.image(bp, caption="Figure 12: Benchmark Comparison", use_container_width=True)

    st.subheader("Diagnostic Assessment")
    st.write(f"**Overfitting Assessment:** {diag['overfit_status']} – {diag['overfit_desc']}")
    st.write(f"**Underfitting Assessment:** {diag['underfit_status']} – {diag['underfit_desc']}")

except Exception as e:
    st.warning(f"Error: {e}")
