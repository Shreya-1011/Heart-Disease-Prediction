import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("model/KNN_heart.pkl")
scaler = joblib.load("model/scaler.pkl")
expected_columns = joblib.load("model/columns.pkl")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.big-title {
    font-size: 42px;
    font-weight: 700;
    color: #ff4b4b;
    text-align: center;
    margin-bottom: 5px;
}

.sub-text {
    text-align: center;
    color: #808080;
    margin-bottom: 30px;
    font-size: 16px;
}

.stButton > button {
    width: 100%;
    background-color: #ff4b4b;
    color: white;
    font-size: 18px;
    border-radius: 12px;
    height: 52px;
    border: none;
    transition: 0.3s;
}

.stButton > button:hover {
    background-color: #ff2e2e;
    color: white;
}

div[data-baseweb="select"] {
    border-radius: 10px;
}

.stNumberInput input {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown(
    '<div class="big-title">❤️ Heart Disease Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-text">Enter your health details for AI-based prediction</div>',
    unsafe_allow_html=True
)

# ---------------- USER INPUT ----------------
col1, col2 = st.columns(2)

with col1:

    age = st.slider(
        "Age",
        18,
        100,
        40
    )

    sex = st.radio(
        "Gender",
        ["Male", "Female"],
        horizontal=True
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=80,
        max_value=200,
        value=120
    )

    cholesterol = st.number_input(
        "Cholesterol (mg/dL)",
        min_value=100,
        max_value=600,
        value=200
    )

    fasting_bs = st.radio(
        "Fasting Blood Sugar > 120 mg/dL",
        [0, 1],
        horizontal=True
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )

with col2:

    max_hr = st.slider(
        "Max Heart Rate",
        60,
        220,
        150
    )

    exercise_angina = st.radio(
        "Exercise-Induced Angina",
        ["Y", "N"],
        horizontal=True
    )

    oldpeak = st.slider(
        "Oldpeak (ST Depression)",
        0.0,
        6.0,
        1.0
    )

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "TA", "ASY"]
    )

    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )

# ---------------- PREDICT BUTTON ----------------
if st.button("Predict Heart Risk"):

    # Create input dictionary
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,

        'Sex_' + ("M" if sex == "Male" else "F"): 1,

        'ChestPainType_' + chest_pain: 1,

        'RestingECG_' + resting_ecg: 1,

        'ExerciseAngina_' + exercise_angina: 1,

        'ST_Slope_' + st_slope: 1
    }

    # Convert into dataframe
    input_df = pd.DataFrame([raw_input])

    # Add missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Arrange columns
    input_df = input_df[expected_columns]

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(scaled_input)[0]

    # Probability
    probability = model.predict_proba(scaled_input)[0][1]

    st.markdown("---")

    # ---------------- RESULT ----------------
    if prediction == 1:

        st.error(
            f"⚠️ High Risk of Heart Disease\n\n"
            f"Risk Score: {probability * 100:.2f}%"
        )

        st.progress(int(probability * 100))

        st.info(
            "💡 Suggestion: Maintain healthy cholesterol, "
            "exercise regularly, and consult a doctor."
        )

    else:

        st.success(
            f"✅ Low Risk of Heart Disease\n\n"
            f"Risk Score: {probability * 100:.2f}%"
        )

        st.progress(int(probability * 100))

        st.info(
            "🎉 Your heart health looks good. "
            "Continue maintaining a healthy lifestyle."
        )