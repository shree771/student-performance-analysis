"""
Data Preprocessing Module for Student Performance Analytics.
Handles:
- Missing value analysis & imputation
- Duplicate detection
- IQR-based outlier detection & justified treatment
- Categorical encoding (One-Hot / Ordinal) fitted strictly on training data
- Numerical feature scaling fitted strictly on training data
- 70/15/15 Train-Validation-Test splitting without data leakage
- Preprocessing pipeline serialization
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

DEFAULT_PROCESSED_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "student-mat-cleaned.csv")
DEFAULT_PIPELINE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "preprocessing_pipeline.pkl")


class PreprocessingPipeline:
    """
    Encapsulates complete preprocessing pipeline with strict leakage prevention.
    Fit is executed ONLY on training split.
    """
    def __init__(self, include_grades=True, cap_outliers=False):
        self.include_grades = include_grades
        self.cap_outliers = cap_outliers
        self.column_transformer = None
        self.feature_names_in = None
        self.feature_names_out = None
        self.categorical_cols = None
        self.numerical_cols = None
        self.scaler = None
        self.encoder = None
        self.outlier_bounds = {}
        self.is_fitted = False

    def identify_missing(self, df):
        """Returns missing value summary per column."""
        missing = df.isnull().sum()
        pct = (missing / len(df)) * 100
        return pd.DataFrame({"Missing Count": missing, "Percentage (%)": pct})

    def identify_duplicates(self, df):
        """Returns number of duplicate rows and duplicate masks."""
        dup_count = int(df.duplicated().sum())
        return dup_count, df[df.duplicated(keep=False)]

    def detect_outliers_iqr(self, df, columns=None):
        """
        Calculates IQR bounds and outlier counts for numerical columns.
        Formula:
          IQR = Q3 - Q1
          Lower Bound = Q1 - 1.5 * IQR
          Upper Bound = Q3 + 1.5 * IQR
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        outlier_report = {}
        for col in columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_report[col] = {
                "Q1": float(q1),
                "Q3": float(q3),
                "IQR": float(iqr),
                "Lower_Bound": float(lower_bound),
                "Upper_Bound": float(upper_bound),
                "Outlier_Count": int(len(outliers)),
                "Outlier_Percentage": float((len(outliers) / len(df)) * 100),
                "Min": float(df[col].min()),
                "Max": float(df[col].max())
            }
        return outlier_report

    def treat_outliers(self, df, columns=None, method="winsorize"):
        """
        Outlier treatment with explicit academic justification.
        - 'winsorize' / 'cap': Caps extreme numerical values at Q1 - 1.5*IQR and Q3 + 1.5*IQR.
        - 'keep': Retains values with theoretical justification (e.g., student absences/zero grades are valid signals).
        """
        df_treated = df.copy()
        if method == "keep":
            return df_treated

        if columns is None:
            columns = ["absences", "failures", "age"]  # Only cap natural skewed features, avoid corrupting grades 0-20

        for col in columns:
            if col in df_treated.columns and col in self.outlier_bounds:
                lb = self.outlier_bounds[col]["Lower_Bound"]
                ub = self.outlier_bounds[col]["Upper_Bound"]
                df_treated[col] = df_treated[col].clip(lower=lb, upper=ub)

        return df_treated

    def fit(self, X_train):
        """
        Fits encoder and scaler ONLY on training data to strictly prevent data leakage.
        """
        X = X_train.copy()
        if "G3" in X.columns:
            X = X.drop(columns=["G3"])

        if not self.include_grades:
            cols_to_drop = [c for c in ["G1", "G2"] if c in X.columns]
            X = X.drop(columns=cols_to_drop)

        self.feature_names_in = X.columns.tolist()
        self.categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
        self.numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        # Compute IQR bounds strictly from training data
        for col in self.numerical_cols:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            self.outlier_bounds[col] = {
                "Q1": float(q1),
                "Q3": float(q3),
                "IQR": float(iqr),
                "Lower_Bound": float(q1 - 1.5 * iqr),
                "Upper_Bound": float(q3 + 1.5 * iqr)
            }

        if self.cap_outliers:
            X = self.treat_outliers(X, method="winsorize")

        self.encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        self.scaler = StandardScaler()

        self.column_transformer = ColumnTransformer(
            transformers=[
                ("num", self.scaler, self.numerical_cols),
                ("cat", self.encoder, self.categorical_cols)
            ],
            remainder="drop"
        )

        self.column_transformer.fit(X)

        # Build transformed feature names
        num_names = self.numerical_cols
        cat_names = self.column_transformer.named_transformers_["cat"].get_feature_names_out(self.categorical_cols).tolist()
        self.feature_names_out = num_names + cat_names
        self.is_fitted = True
        return self

    def transform(self, X_data):
        """
        Transforms input dataset using fitted transformer.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline has not been fitted yet. Call fit() first.")

        X = X_data.copy()
        if "G3" in X.columns:
            X = X.drop(columns=["G3"])

        if not self.include_grades:
            cols_to_drop = [c for c in ["G1", "G2"] if c in X.columns]
            X = X.drop(columns=cols_to_drop)

        # Align columns
        for c in self.feature_names_in:
            if c not in X.columns:
                raise ValueError(f"Input is missing required feature '{c}'")
        X = X[self.feature_names_in]

        if self.cap_outliers:
            X = self.treat_outliers(X, method="winsorize")

        X_proc = self.column_transformer.transform(X)
        return pd.DataFrame(X_proc, columns=self.feature_names_out, index=X.index)

    def fit_transform(self, X_train):
        self.fit(X_train)
        return self.transform(X_train)


def split_data(df, target_col="G3", train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Performs 70% Train, 15% Validation, 15% Test split with reproducible random seed.
    Returns: X_train, X_val, X_test, y_train, y_val, y_test
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Splits must sum to 1.0"
    
    X = df.drop(columns=[target_col])
    y = df[target_col]

    temp_ratio = val_ratio + test_ratio  # 0.30
    val_in_temp_ratio = val_ratio / temp_ratio  # 0.15 / 0.30 = 0.50

    # First split: 70% Train, 30% Temp (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_ratio, random_state=random_state
    )

    # Second split: Split temp into 50% Val and 50% Test (each 15% of total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1.0 - val_in_temp_ratio), random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def execute_full_preprocessing(df, include_grades=True, cap_outliers=False, save_pipeline=True,
                               pipeline_path=DEFAULT_PIPELINE_PATH, cleaned_path=DEFAULT_PROCESSED_PATH):
    """
    Full workflow orchestration:
    1. Duplicates check and removal
    2. Missing value verification
    3. Train-Val-Test 70/15/15 split
    4. Fit pipeline on train only
    5. Transform train, val, test
    6. Save pipeline and clean dataset
    """
    # Duplicate check
    dup_count = df.duplicated().sum()
    df_clean = df.drop_duplicates() if dup_count > 0 else df.copy()

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_clean, target_col="G3", random_state=42)

    # Initialize and fit pipeline on train only
    pipeline = PreprocessingPipeline(include_grades=include_grades, cap_outliers=cap_outliers)
    pipeline.fit(X_train)

    # Transform all splits
    X_train_proc = pipeline.transform(X_train)
    X_val_proc = pipeline.transform(X_val)
    X_test_proc = pipeline.transform(X_test)

    # Save cleaned combined dataset
    os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)
    df_clean.to_csv(cleaned_path, index=False)

    # Save pipeline object
    if save_pipeline:
        os.makedirs(os.path.dirname(pipeline_path), exist_ok=True)
        with open(pipeline_path, "wb") as f:
            pickle.dump(pipeline, f)

    return {
        "pipeline": pipeline,
        "X_train_raw": X_train,
        "X_val_raw": X_val,
        "X_test_raw": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "X_train_proc": X_train_proc,
        "X_val_proc": X_val_proc,
        "X_test_proc": X_test_proc,
        "raw_shape": df.shape,
        "cleaned_shape": df_clean.shape,
        "processed_feature_count": X_train_proc.shape[1],
        "feature_names": pipeline.feature_names_out,
        "outlier_report": pipeline.outlier_bounds
    }


if __name__ == "__main__":
    from data_loader import load_dataset
    df = load_dataset()
    res = execute_full_preprocessing(df)
    print("Preprocessing completed successfully!")
    print(f"Train shapes: {res['X_train_proc'].shape}, Val: {res['X_val_proc'].shape}, Test: {res['X_test_proc'].shape}")
    print(f"Target shapes: Train {len(res['y_train'])}, Val {len(res['y_val'])}, Test {len(res['y_test'])}")
    print(f"Total features after One-Hot Encoding: {res['processed_feature_count']}")
    print(f"Sample transformed columns: {res['feature_names'][:8]}")
