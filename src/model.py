"""
Deep Learning (MLP) and Baseline Modeling Module for Student Performance Analytics.
Implements:
- Keras Multi-Layer Perceptron (MLP) Regression Architecture:
  * Input layer -> Dense(128, ReLU) -> Dropout(0.2)
  * Dense(64, ReLU) -> Dropout(0.2)
  * Dense(32, ReLU) -> Output Dense(1, Linear)
- Adam optimizer, MSE loss, MAE metric, EarlyStopping, ModelCheckpoint
- Hyperparameter tuning across Learning Rates, Dropout Rates, and Architectures
- Baseline Models: Linear Regression & Random Forest Regressor
- Early Prediction Experiment (Without G1 & G2) vs Complete Model (With G1 & G2)
- Model persistence to models/best_model.keras
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers, callbacks
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "results")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.keras")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def build_mlp_model(input_dim, hidden_layers=[128, 64, 32], dropout_rate=0.2, learning_rate=0.001):
    """
    Builds and compiles Keras Multi-Layer Perceptron (MLP) Regressor.
    """
    model = keras.Sequential(name="StudentPerformance_MLP")
    model.add(layers.Input(shape=(input_dim,)))

    # Hidden Layer 1
    model.add(layers.Dense(hidden_layers[0], activation="relu", name="dense_hidden_1"))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate, name="dropout_1"))

    # Hidden Layer 2
    model.add(layers.Dense(hidden_layers[1], activation="relu", name="dense_hidden_2"))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate, name="dropout_2"))

    # Hidden Layer 3
    model.add(layers.Dense(hidden_layers[2], activation="relu", name="dense_hidden_3"))

    # Output Layer
    model.add(layers.Dense(1, activation="linear", name="grade_output"))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mae"])
    return model


def train_mlp_model(X_train, y_train, X_val, y_val,
                    hidden_layers=[128, 64, 32],
                    dropout_rate=0.2,
                    learning_rate=0.001,
                    epochs=100,
                    batch_size=32,
                    model_save_path=BEST_MODEL_PATH):
    """
    Trains MLP model with EarlyStopping and ModelCheckpoint.
    Monitors validation loss.
    """
    input_dim = X_train.shape[1]
    model = build_mlp_model(input_dim, hidden_layers, dropout_rate, learning_rate)

    cb_early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    cb_checkpoint = callbacks.ModelCheckpoint(
        filepath=model_save_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=0
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[cb_early_stop, cb_checkpoint],
        verbose=1
    )

    return model, history


def run_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    """
    Compares different learning rates, hidden layer architectures, and dropout rates
    using the validation dataset.
    """
    experiments = [
        {"name": "Config 1 (Standard 128-64-32, lr=0.001, drop=0.2)", "layers": [128, 64, 32], "lr": 0.001, "dropout": 0.2},
        {"name": "Config 2 (Smaller 64-32-16, lr=0.001, drop=0.1)", "layers": [64, 32, 16], "lr": 0.001, "dropout": 0.1},
        {"name": "Config 3 (Larger 256-128-64, lr=0.0005, drop=0.3)", "layers": [256, 128, 64], "lr": 0.0005, "dropout": 0.3},
        {"name": "Config 4 (Standard 128-64-32, lr=0.0005, drop=0.2)", "layers": [128, 64, 32], "lr": 0.0005, "dropout": 0.2},
        {"name": "Config 5 (Standard 128-64-32, lr=0.005, drop=0.2)", "layers": [128, 64, 32], "lr": 0.005, "dropout": 0.2}
    ]

    results = []
    best_config = None
    best_val_mae = float("inf")

    print("Running hyperparameter tuning across 5 configurations...")
    for exp in experiments:
        temp_model = build_mlp_model(X_train.shape[1], exp["layers"], exp["dropout"], exp["lr"])
        cb = callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)
        hist = temp_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=70,
            batch_size=32,
            callbacks=[cb],
            verbose=0
        )
        val_preds = temp_model.predict(X_val, verbose=0).flatten()
        val_mae = mean_absolute_error(y_val, val_preds)
        val_mse = mean_squared_error(y_val, val_preds)
        val_rmse = np.sqrt(val_mse)
        val_r2 = r2_score(y_val, val_preds)

        record = {
            "name": exp["name"],
            "layers": str(exp["layers"]),
            "learning_rate": exp["lr"],
            "dropout": exp["dropout"],
            "val_mae": round(float(val_mae), 4),
            "val_mse": round(float(val_mse), 4),
            "val_rmse": round(float(val_rmse), 4),
            "val_r2": round(float(val_r2), 4),
            "stopped_epoch": len(hist.history["loss"])
        }
        results.append(record)
        print(f"  {exp['name']} -> Val MAE: {val_mae:.4f}, Val R²: {val_r2:.4f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_config = exp

    return pd.DataFrame(results), best_config


def train_baseline_models(X_train, y_train, X_test, y_test):
    """
    Fits and evaluates Linear Regression and Random Forest Regressor on the test set.
    """
    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_metrics = {
        "MAE": round(float(mean_absolute_error(y_test, lr_preds)), 4),
        "MSE": round(float(mean_squared_error(y_test, lr_preds)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, lr_preds))), 4),
        "R2": round(float(r2_score(y_test, lr_preds)), 4)
    }

    # 2. Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=6)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_metrics = {
        "MAE": round(float(mean_absolute_error(y_test, rf_preds)), 4),
        "MSE": round(float(mean_squared_error(y_test, rf_preds)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, rf_preds))), 4),
        "R2": round(float(r2_score(y_test, rf_preds)), 4)
    }

    return {
        "Linear Regression": {"model": lr, "metrics": lr_metrics, "preds": lr_preds},
        "Random Forest": {"model": rf, "metrics": rf_metrics, "preds": rf_preds}
    }


def run_full_training_pipeline(prep_res):
    """
    Executes:
    1. Hyperparameter tuning on validation split
    2. Final training of best MLP model on training split with checkpointing
    3. Baseline training (Linear Regression, Random Forest)
    4. Early prediction experiment (training without G1 and G2)
    5. Saves metrics & history to outputs/results/model_results.json
    """
    X_train = prep_res["X_train_proc"]
    y_train = prep_res["y_train"]
    X_val = prep_res["X_val_proc"]
    y_val = prep_res["y_val"]
    X_test = prep_res["X_test_proc"]
    y_test = prep_res["y_test"]

    # 1. Hyperparameter Tuning
    tuning_df, best_cfg = run_hyperparameter_tuning(X_train, y_train, X_val, y_val)

    # 2. Train Primary MLP Model with configured architecture
    print("\nTraining Primary Deep Learning MLP Model...")
    mlp_model, history = train_mlp_model(
        X_train, y_train, X_val, y_val,
        hidden_layers=[128, 64, 32],
        dropout_rate=0.2,
        learning_rate=0.001,
        epochs=100,
        batch_size=32,
        model_save_path=BEST_MODEL_PATH
    )

    # Final test evaluation of best MLP
    best_loaded_model = keras.models.load_model(BEST_MODEL_PATH)
    test_preds = best_loaded_model.predict(X_test).flatten()
    mlp_test_metrics = {
        "MAE": round(float(mean_absolute_error(y_test, test_preds)), 4),
        "MSE": round(float(mean_squared_error(y_test, test_preds)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, test_preds))), 4),
        "R2": round(float(r2_score(y_test, test_preds)), 4)
    }
    print(f"MLP Test Set Metrics: MAE={mlp_test_metrics['MAE']}, RMSE={mlp_test_metrics['RMSE']}, R²={mlp_test_metrics['R2']}")

    # 3. Train Baselines
    print("\nTraining Baseline Models...")
    baselines = train_baseline_models(X_train, y_train, X_test, y_test)

    # 4. Early Prediction Experiment (Without G1 and G2)
    print("\nRunning Early Prediction Experiment (Excluding G1 and G2)...")
    from preprocessing import PreprocessingPipeline
    early_pipe = PreprocessingPipeline(include_grades=False)
    X_train_early = early_pipe.fit_transform(prep_res["X_train_raw"])
    X_val_early = early_pipe.transform(prep_res["X_val_raw"])
    X_test_early = early_pipe.transform(prep_res["X_test_raw"])

    early_model = build_mlp_model(X_train_early.shape[1], [128, 64, 32], 0.2, 0.001)
    early_cb = callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    early_hist = early_model.fit(
        X_train_early, y_train,
        validation_data=(X_val_early, y_val),
        epochs=100, batch_size=32,
        callbacks=[early_cb], verbose=0
    )
    early_test_preds = early_model.predict(X_test_early).flatten()
    early_metrics = {
        "MAE": round(float(mean_absolute_error(y_test, early_test_preds)), 4),
        "MSE": round(float(mean_squared_error(y_test, early_test_preds)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, early_test_preds))), 4),
        "R2": round(float(r2_score(y_test, early_test_preds)), 4)
    }
    print(f"Early Prediction (No G1/G2) Test Metrics: MAE={early_metrics['MAE']}, RMSE={early_metrics['RMSE']}, R²={early_metrics['R2']}")

    # 5. Compile and save results
    history_dict = {
        "loss": [float(x) for x in history.history["loss"]],
        "val_loss": [float(x) for x in history.history["val_loss"]],
        "mae": [float(x) for x in history.history["mae"]],
        "val_mae": [float(x) for x in history.history["val_mae"]],
        "epochs": len(history.history["loss"])
    }

    results_data = {
        "mlp_test_metrics": mlp_test_metrics,
        "baseline_metrics": {
            "Linear_Regression": baselines["Linear Regression"]["metrics"],
            "Random_Forest": baselines["Random Forest"]["metrics"]
        },
        "early_prediction_metrics": early_metrics,
        "tuning_results": tuning_df.to_dict(orient="records"),
        "best_hyperparameters": best_cfg,
        "history": history_dict,
        "test_actual": [float(x) for x in y_test.values],
        "test_predicted_mlp": [float(x) for x in test_preds],
        "test_predicted_rf": [float(x) for x in baselines["Random Forest"]["preds"]],
        "test_predicted_lr": [float(x) for x in baselines["Linear Regression"]["preds"]],
        "early_test_predicted": [float(x) for x in early_test_preds]
    }

    results_path = os.path.join(RESULTS_DIR, "model_results.json")
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=4)
    print(f"Model results saved successfully to {results_path}")

    return results_data, best_loaded_model


if __name__ == "__main__":
    from data_loader import load_dataset
    from preprocessing import execute_full_preprocessing
    df = load_dataset()
    prep = execute_full_preprocessing(df)
    results, model = run_full_training_pipeline(prep)
    print("\nTraining Pipeline Finished.")
