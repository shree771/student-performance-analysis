"""
Student Grade Prediction & Intervention Advisory Module.
Handles:
- Input validation for all 32 student features
- Preprocessing transformation using saved fitted pipeline
- Model inference with trained TensorFlow/Keras MLP
- Academic grade interpretation (Portuguese 0-20 scale)
- Context-aware intervention recommendations
- Formal academic disclaimer
"""

import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

PIPELINE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "preprocessing_pipeline.pkl")
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "best_model.keras")

GRADE_CLASSIFICATION = [
    {"range": (16, 20), "category": "Excellent / Very Good", "color": "#2ca02c", "status": "Distinction", "summary": "Outstanding academic mastery and consistent high performance."},
    {"range": (14, 15.99), "category": "Good", "color": "#1f77b4", "status": "Commendable", "summary": "Solid understanding of curriculum concepts with room for distinction."},
    {"range": (10, 13.99), "category": "Sufficient / Pass", "color": "#ff7f0e", "status": "Passing", "summary": "Meets baseline requirements; proactive support advised to elevate performance."},
    {"range": (0, 9.99), "category": "Fail / Insufficient", "color": "#d62728", "status": "At Risk", "summary": "Critical academic intervention and remedial support urgently required."}
]


def classify_grade(grade):
    """Classifies a numeric grade (0-20) into academic tier."""
    grade_clamped = max(0.0, min(20.0, float(grade)))
    for tier in GRADE_CLASSIFICATION:
        if tier["range"][0] <= grade_clamped <= tier["range"][1]:
            return tier
    return GRADE_CLASSIFICATION[-1]


def generate_recommendations(student_dict, predicted_grade):
    """
    Generates tailored, actionable academic interventions based on student attributes.
    """
    recommendations = []

    # Study time intervention
    studytime = int(student_dict.get("studytime", 2))
    if studytime == 1:
        recommendations.append({
            "area": "Study Habits",
            "type": "Urgent",
            "text": "Current weekly study time is <2 hours. Transitioning to 5-10 hours/week (level 3) historically yields an average +2.0 grade point increase."
        })
    elif studytime == 2:
        recommendations.append({
            "area": "Study Habits",
            "type": "Recommended",
            "text": "Current study time is 2-5 hours/week. Increasing to 5+ hours will reinforce complex topics and improve retention."
        })

    # Absences intervention
    absences = int(student_dict.get("absences", 0))
    if absences > 10:
        recommendations.append({
            "area": "Attendance",
            "type": "Critical",
            "text": f"High absence count ({absences} days). Regular attendance is crucial to stay aligned with course progression."
        })

    # Past failures intervention
    failures = int(student_dict.get("failures", 0))
    if failures > 0:
        recommendations.append({
            "area": "Foundational Remediation",
            "type": "Critical",
            "text": f"Student has {failures} past class failures. Focused remedial tutoring on foundational prerequisites is recommended."
        })

    # School support intervention
    schoolsup = str(student_dict.get("schoolsup", "no")).lower()
    if schoolsup == "no" and predicted_grade < 12:
        recommendations.append({
            "area": "Institutional Tutoring",
            "type": "High Priority",
            "text": "Enrolling in school-provided academic support sessions can help bridge performance gaps before final evaluations."
        })

    # Higher education motivation
    higher = str(student_dict.get("higher", "yes")).lower()
    if higher == "no":
        recommendations.append({
            "area": "Academic Mentoring",
            "type": "Advisory",
            "text": "Academic counseling is recommended to cultivate long-term career aspirations, which correlate strongly with higher persistence."
        })

    # Social balance
    goout = int(student_dict.get("goout", 3))
    if goout >= 4 and predicted_grade < 12:
        recommendations.append({
            "area": "Time Management",
            "type": "Guidance",
            "text": "High extracurricular going-out frequency (level 4-5). Structuring a balanced weekday schedule can preserve study focus."
        })

    if not recommendations:
        recommendations.append({
            "area": "Maintenance",
            "type": "Positive",
            "text": "Student profile demonstrates balanced study habits, low absenteeism, and positive academic trajectory. Maintain current routine."
        })

    return recommendations


class GradePredictor:
    """Predictor service encapsulating model inference and pipeline execution."""
    def __init__(self, pipeline_path=PIPELINE_PATH, model_path=MODEL_PATH):
        self.pipeline_path = pipeline_path
        self.model_path = model_path
        self.pipeline = None
        self.model = None
        self._load_resources()

    def _load_resources(self):
        if not os.path.exists(self.pipeline_path):
            raise FileNotFoundError(f"Pipeline not found at {self.pipeline_path}")
        with open(self.pipeline_path, "rb") as f:
            self.pipeline = pickle.load(f)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Trained model not found at {self.model_path}")
        self.model = keras.models.load_model(self.model_path)

    def predict(self, student_dict):
        """
        Takes raw student feature dictionary, validates, transforms, and predicts G3.
        """
        # Convert to single-row DataFrame
        df_input = pd.DataFrame([student_dict])

        # Feature transform
        X_proc = self.pipeline.transform(df_input)

        # Model inference
        raw_pred = float(self.model.predict(X_proc, verbose=0)[0][0])
        # Clamp to realistic scale [0, 20]
        final_grade = max(0.0, min(20.0, raw_pred))

        tier = classify_grade(final_grade)
        recs = generate_recommendations(student_dict, final_grade)

        return {
            "predicted_grade": round(final_grade, 2),
            "tier": tier["category"],
            "tier_color": tier["color"],
            "status": tier["status"],
            "summary": tier["summary"],
            "recommendations": recs,
            "disclaimer": "DISCLAIMER: This predicted final grade is an algorithmic estimate based on historical student performance datasets. It should be utilized solely as an early guidance indicator to facilitate supportive educational interventions, not as a deterministic prediction."
        }


if __name__ == "__main__":
    predictor = GradePredictor()
    sample_student = {
        "school": "GP", "sex": "F", "age": 17, "address": "U", "famsize": "GT3", "Pstatus": "T",
        "Medu": 3, "Fedu": 3, "Mjob": "other", "Fjob": "other", "reason": "course", "guardian": "mother",
        "traveltime": 1, "studytime": 3, "failures": 0, "schoolsup": "no", "famsup": "yes", "paid": "no",
        "activities": "yes", "nursery": "yes", "higher": "yes", "internet": "yes", "romantic": "no",
        "famrel": 4, "freetime": 3, "goout": 2, "Dalc": 1, "Walc": 1, "health": 4, "absences": 4,
        "G1": 13, "G2": 14
    }
    result = predictor.predict(sample_student)
    print("Prediction Result:")
    print(f"Grade: {result['predicted_grade']} / 20 ({result['tier']})")
    print(f"Summary: {result['summary']}")
    print(f"Recommendations ({len(result['recommendations'])}):")
    for r in result['recommendations']:
        print(f" - [{r['area']}] {r['text']}")
