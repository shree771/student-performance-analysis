"""
Data Loader Module for Student Performance Analytics.
Automatically downloads and validates the UCI Student Performance Dataset (student-mat.csv).
"""

import os
import urllib.request
import zipfile
import io
import pandas as pd

UCI_DATASET_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
DEFAULT_DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "student-mat.csv")
RAW_DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "student-mat-raw.csv")

ATTRIBUTE_METADATA = {
    "school": {"type": "binary", "description": "Student's school ('GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)"},
    "sex": {"type": "binary", "description": "Student's sex ('F' - female or 'M' - male)"},
    "age": {"type": "numeric", "description": "Student's age (from 15 to 22)"},
    "address": {"type": "binary", "description": "Student's home address type ('U' - urban or 'R' - rural)"},
    "famsize": {"type": "binary", "description": "Family size ('LE3' - less or equal to 3 or 'GT3' - greater than 3)"},
    "Pstatus": {"type": "binary", "description": "Parent's cohabitation status ('T' - living together or 'A' - apart)"},
    "Medu": {"type": "numeric/ordinal", "description": "Mother's education (0 - none, 1 - primary, 2 - 5th to 9th grade, 3 - secondary, 4 - higher)"},
    "Fedu": {"type": "numeric/ordinal", "description": "Father's education (0 - none, 1 - primary, 2 - 5th to 9th grade, 3 - secondary, 4 - higher)"},
    "Mjob": {"type": "nominal", "description": "Mother's job ('teacher', 'health' care, civil 'services', 'at_home' or 'other')"},
    "Fjob": {"type": "nominal", "description": "Father's job ('teacher', 'health' care, civil 'services', 'at_home' or 'other')"},
    "reason": {"type": "nominal", "description": "Reason to choose this school ('home' close, school 'reputation', 'course' preference or 'other')"},
    "guardian": {"type": "nominal", "description": "Student's guardian ('mother', 'father' or 'other')"},
    "traveltime": {"type": "numeric/ordinal", "description": "Home to school travel time (1 - <15 min, 2 - 15-30 min, 3 - 30 min-1 hour, 4 - >1 hour)"},
    "studytime": {"type": "numeric/ordinal", "description": "Weekly study time (1 - <2 hours, 2 - 2-5 hours, 3 - 5-10 hours, 4 - >10 hours)"},
    "failures": {"type": "numeric", "description": "Number of past class failures (n if 1<=n<3, else 4)"},
    "schoolsup": {"type": "binary", "description": "Extra educational support (yes or no)"},
    "famsup": {"type": "binary", "description": "Family educational support (yes or no)"},
    "paid": {"type": "binary", "description": "Extra paid classes within the course subject (yes or no)"},
    "activities": {"type": "binary", "description": "Extra-curricular activities (yes or no)"},
    "nursery": {"type": "binary", "description": "Attended nursery school (yes or no)"},
    "higher": {"type": "binary", "description": "Wants to take higher education (yes or no)"},
    "internet": {"type": "binary", "description": "Internet access at home (yes or no)"},
    "romantic": {"type": "binary", "description": "With a romantic relationship (yes or no)"},
    "famrel": {"type": "numeric/ordinal", "description": "Quality of family relationships (from 1 - very bad to 5 - excellent)"},
    "freetime": {"type": "numeric/ordinal", "description": "Free time after school (from 1 - very low to 5 - very high)"},
    "goout": {"type": "numeric/ordinal", "description": "Going out with friends (from 1 - very low to 5 - very high)"},
    "Dalc": {"type": "numeric/ordinal", "description": "Workday alcohol consumption (from 1 - very low to 5 - very high)"},
    "Walc": {"type": "numeric/ordinal", "description": "Weekend alcohol consumption (from 1 - very low to 5 - very high)"},
    "health": {"type": "numeric/ordinal", "description": "Current health status (from 1 - very bad to 5 - very good)"},
    "absences": {"type": "numeric", "description": "Number of school absences (from 0 to 93)"},
    "G1": {"type": "numeric", "description": "First period grade (from 0 to 20)"},
    "G2": {"type": "numeric", "description": "Second period grade (from 0 to 20)"},
    "G3": {"type": "target/numeric", "description": "Final grade (from 0 to 20) - Primary Regression Target"}
}


def download_uci_dataset(target_path=DEFAULT_DATASET_PATH, force=False):
    """
    Downloads student performance zip from UCI, extracts student-mat.csv,
    and preserves both original raw copy and working dataset copy.
    """
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    raw_path = RAW_DATASET_PATH

    if os.path.exists(target_path) and not force:
        try:
            df = pd.read_csv(target_path, sep=';')
            if df.shape == (395, 33):
                return True, f"Dataset already exists and verified at {target_path} (Shape: {df.shape})"
        except Exception:
            pass

    try:
        req = urllib.request.Request(UCI_DATASET_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            outer_zip = zipfile.ZipFile(io.BytesIO(response.read()))
            mat_content = None

            # Check if student-mat.csv is directly in the outer zip or inside student.zip
            for name in outer_zip.namelist():
                if name.endswith("student-mat.csv"):
                    mat_content = outer_zip.read(name)
                    break
                elif name.endswith("student.zip"):
                    inner_zip = zipfile.ZipFile(io.BytesIO(outer_zip.read(name)))
                    for inner_name in inner_zip.namelist():
                        if inner_name.endswith("student-mat.csv"):
                            mat_content = inner_zip.read(inner_name)
                            break

            if mat_content is None:
                return False, "Could not locate student-mat.csv inside UCI archive zip."

            # Write working copy and raw copy
            with open(target_path, "wb") as f:
                f.write(mat_content)
            with open(raw_path, "wb") as f:
                f.write(mat_content)

            # Verification
            df = pd.read_csv(target_path, sep=';')
            if df.shape == (395, 33):
                return True, f"Successfully downloaded and verified UCI dataset: {df.shape[0]} records, {df.shape[1]} attributes."
            else:
                return False, f"Downloaded dataset has unexpected shape: {df.shape} (expected 395, 33)."

    except Exception as e:
        return False, f"Failed to download dataset from UCI: {str(e)}"


def load_dataset(filepath=DEFAULT_DATASET_PATH):
    """
    Loads student performance dataset. If not found, attempts auto-download.
    Returns pandas DataFrame.
    """
    if not os.path.exists(filepath):
        success, msg = download_uci_dataset(filepath)
        if not success:
            raise FileNotFoundError(f"Dataset not found and auto-download failed: {msg}")

    # The UCI student dataset uses semicolon delimiter
    df = pd.read_csv(filepath, sep=';')
    return df


def get_dataset_summary(df):
    """
    Returns structured verification metadata for the dataset.
    """
    summary = {
        "num_records": int(df.shape[0]),
        "num_attributes": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "target_variable": "G3",
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "first_5_records": df.head(5).to_dict(orient="records"),
        "numeric_features": df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        "categorical_features": df.select_dtypes(include=['object']).columns.tolist()
    }
    return summary


if __name__ == "__main__":
    ok, message = download_uci_dataset()
    print("Download status:", ok)
    print("Message:", message)
    df = load_dataset()
    summary = get_dataset_summary(df)
    print(f"Loaded records: {summary['num_records']}, features: {summary['num_attributes']}")
    print(f"Target column G3 present: {'G3' in summary['columns']}")
    print("Sample G3 values:", df['G3'].describe().to_dict())
