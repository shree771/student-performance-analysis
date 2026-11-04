"""Page 6: Deep Learning Model"""
import os
import sys
import streamlit as st
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from evaluation import load_model_results

st.set_page_config(page_title="Deep Learning Model", page_icon="🧠", layout="wide")
st.title("🧠 Deep Learning Model (MLP Regressor)")

st.subheader("Neural Network Architecture")
st.markdown("""
```text
[Input Layer: 41 Features]
         │
         ▼
[Hidden Layer 1: Dense(128, ReLU)] -> [Dropout(0.2)]
         │
         ▼
[Hidden Layer 2: Dense(64, ReLU)]  -> [Dropout(0.2)]
         │
         ▼
[Hidden Layer 3: Dense(32, ReLU)]
         │
         ▼
[Output Layer: Dense(1, Linear)] -> Predicted Grade G3 (0 to 20)
```
""")

try:
    results = load_model_results()
    st.subheader("Validation Hyperparameter Tuning Comparison")
    st.dataframe(pd.DataFrame(results["tuning_results"]), use_container_width=True)
    st.success(f"Optimal Hyperparameter Configuration: {results['best_hyperparameters']['name']}")

    st.markdown("---")
    st.subheader("Early Prediction Experiment (Excluding G1 & G2)")
    e = results["early_prediction_metrics"]
    m = results["mlp_test_metrics"]
    c1, c2 = st.columns(2)
    c1.metric("Early Prediction (No G1/G2) Test MAE", f"{e['MAE']:.3f}", f"R² = {e['R2']:.3f}")
    c2.metric("Complete Model (With G1/G2) Test MAE", f"{m['MAE']:.3f}", f"R² = {m['R2']:.3f}")
except Exception as e:
    st.warning(f"Results not available: {e}. Run `python src/model.py`.")
