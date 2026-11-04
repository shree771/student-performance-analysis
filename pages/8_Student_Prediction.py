"""Page 8: Student Prediction"""
import os
import sys
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from prediction import GradePredictor

st.set_page_config(page_title="Student Prediction", page_icon="🔮", layout="wide")
st.title("🔮 Interactive Student Grade Prediction")
st.caption("Enter student profile attributes to predict final mathematics grade (0-20) and receive actionable interventions.")

predictor = GradePredictor()

with st.form("student_form_page"):
    c1, c2, c3, c4 = st.columns(4)
    school = c1.selectbox("School", ["GP", "MS"])
    sex = c2.selectbox("Sex", ["F", "M"])
    age = c3.slider("Age", 15, 22, 17)
    address = c4.selectbox("Address", ["U", "R"])

    c5, c6, c7, c8 = st.columns(4)
    famsize = c5.selectbox("Family Size", ["GT3", "LE3"])
    pstatus = c6.selectbox("Parent Cohabitation", ["T", "A"])
    medu = c7.selectbox("Mother Education (0-4)", [0, 1, 2, 3, 4], index=3)
    fedu = c8.selectbox("Father Education (0-4)", [0, 1, 2, 3, 4], index=3)

    c9, c10, c11, c12 = st.columns(4)
    mjob = c9.selectbox("Mother Job", ["at_home", "health", "other", "services", "teacher"])
    fjob = c10.selectbox("Father Job", ["teacher", "other", "services", "health", "at_home"])
    reason = c11.selectbox("Reason", ["home", "reputation", "course", "other"])
    guardian = c12.selectbox("Guardian", ["mother", "father", "other"])

    c13, c14, c15, c16 = st.columns(4)
    traveltime = c13.selectbox("Travel Time (1-4)", [1, 2, 3, 4], index=0)
    studytime = c14.selectbox("Study Time (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)", [1, 2, 3, 4], index=1)
    failures = c15.selectbox("Past Failures", [0, 1, 2, 3], index=0)
    schoolsup = c16.selectbox("School Support", ["no", "yes"])

    c17, c18, c19, c20 = st.columns(4)
    famsup = c17.selectbox("Family Support", ["yes", "no"])
    paid = c18.selectbox("Paid Classes", ["no", "yes"])
    activities = c19.selectbox("Activities", ["yes", "no"])
    nursery = c20.selectbox("Nursery", ["yes", "no"])

    c21, c22, c23, c24 = st.columns(4)
    higher = c21.selectbox("Higher Education Aspiration", ["yes", "no"])
    internet = c22.selectbox("Internet Access", ["yes", "no"])
    romantic = c23.selectbox("Romantic Relationship", ["no", "yes"])
    famrel = c24.slider("Family Relationship (1-5)", 1, 5, 4)

    c25, c26, c27, c28 = st.columns(4)
    freetime = c25.slider("Free Time (1-5)", 1, 5, 3)
    goout = c26.slider("Going Out (1-5)", 1, 5, 2)
    dalc = c27.slider("Workday Alcohol (1-5)", 1, 5, 1)
    walc = c28.slider("Weekend Alcohol (1-5)", 1, 5, 1)

    c29, c30, c31, c32 = st.columns(4)
    health = c29.slider("Health Status (1-5)", 1, 5, 5)
    absences = c30.number_input("Absences", min_value=0, max_value=93, value=4)
    g1 = c31.slider("Period 1 Grade (G1)", 0, 20, 12)
    g2 = c32.slider("Period 2 Grade (G2)", 0, 20, 13)

    submitted = st.form_submit_button("Predict Final Grade")

if submitted:
    student_dict = {
        "school": school, "sex": sex, "age": age, "address": address, "famsize": famsize,
        "Pstatus": pstatus, "Medu": medu, "Fedu": fedu, "Mjob": mjob, "Fjob": fjob,
        "reason": reason, "guardian": guardian, "traveltime": traveltime, "studytime": studytime,
        "failures": failures, "schoolsup": schoolsup, "famsup": famsup, "paid": paid,
        "activities": activities, "nursery": nursery, "higher": higher, "internet": internet,
        "romantic": romantic, "famrel": famrel, "freetime": freetime, "goout": goout,
        "Dalc": dalc, "Walc": walc, "health": health, "absences": absences, "G1": g1, "G2": g2
    }
    res = predictor.predict(student_dict)
    st.success(f"### Predicted Final Grade: {res['predicted_grade']} / 20 ({res['tier']})")
    st.write(f"**Academic Summary:** {res['summary']}")
    st.write("**Intervention Recommendations:**")
    for r in res["recommendations"]:
        st.write(f"- **[{r['area']}]** {r['text']}")
    st.caption(res["disclaimer"])
