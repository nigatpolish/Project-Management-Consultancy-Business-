import streamlit as st
import pandas as pd
import joblib
from fpdf import FPDF
import datetime
from pathlib import Path
from textwrap import dedent


# ============================================================
# PAGE CONFIGURATION
# IMPORTANT: This must be the first Streamlit command.
# ============================================================

st.set_page_config(
    page_title="Clinical Diabetes Risk Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================

header_left, header_right = st.columns([4, 1])

with header_left:
    st.title("🩺 Clinical Diabetes Risk Prediction Tool")
    st.caption("Data-driven health risk assessment and clinical decision support")

with header_right:
    st.info("🔒 Secure Clinical Assessment")

st.divider()



# ============================================================
# VISUAL MEDICAL HEADER
# ============================================================

HEADER_IMAGE = Path(__file__).resolve().parent / "diabetes_medical_header.png"

if HEADER_IMAGE.exists():
    st.image(
        str(HEADER_IMAGE),
        use_container_width=True,
    )


# ============================================================
# MODEL LOADING
# ============================================================

MODEL_PATH = Path(__file__).resolve().parent / "diabetes_risk_model.pkl"


@st.cache_resource
def load_model(model_path):
    """Load and cache the trained diabetes-risk pipeline."""
    if not model_path.exists():
        return None
    return joblib.load(model_path)


model = load_model(MODEL_PATH)

if model is None:
    st.error(
        f"Model file not found at: `{MODEL_PATH}`. "
        "Please place `diabetes_risk_model.pkl` in the same folder as `app.py`."
    )
    st.stop()



# ============================================================
# APPLICATION INTRODUCTION
# ============================================================

st.markdown(
    "Enter the patient's clinical, demographic, and lifestyle metrics "
    "below to evaluate their model-based diabetes risk and generate a report."
)


# ============================================================
# INPUT FORM
# ============================================================

with st.form("prediction_form"):

    st.subheader("Patient Clinical & Lifestyle Profile")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider(
            "Age (Years)",
            min_value=18,
            max_value=80,
            value=45,
        )

        gender = st.selectbox(
            "Gender",
            ["Female", "Male", "Other"],
        )

        city = st.selectbox(
            "City",
            [
                "Mumbai",
                "Thane",
                "Delhi",
                "Bangalore",
                "Chennai",
                "Kolkata",
                "Hyderabad",
                "Pune",
                "Ahmedabad",
                "Surat",
                "Jaipur",
                "Lucknow",
                "Kanpur",
                "Nagpur",
                "Indore",
                "Bhopal",
                "Visakhapatnam",
                "Patna",
            ],
        )

        bmi = st.slider(
            "BMI (kg/m²)",
            min_value=15.0,
            max_value=41.0,
            value=25.0,
            step=0.1,
        )

        waist_circumference_cm = st.slider(
            "Waist Circumference (cm)",
            min_value=70.0,
            max_value=150.0,
            value=90.0,
            step=0.5,
        )

        income_bracket = st.selectbox(
            "Income Bracket",
            ["Low", "Middle", "High"],
        )

    with col2:
        fasting_blood_sugar = st.slider(
            "Fasting Blood Sugar (mg/dL)",
            min_value=70,
            max_value=400,
            value=100,
        )

        hba1c_level = st.slider(
            "HbA1c Level (%)",
            min_value=4.5,
            max_value=14.0,
            value=5.7,
            step=0.1,
        )

        blood_pressure_systolic = st.slider(
            "Systolic BP (mmHg)",
            min_value=80,
            max_value=200,
            value=120,
        )

        blood_pressure_diastolic = st.slider(
            "Diastolic BP (mmHg)",
            min_value=50,
            max_value=120,
            value=80,
        )

        hours_sleep_per_night = st.slider(
            "Sleep per Night (Hours)",
            min_value=3.0,
            max_value=12.0,
            value=7.0,
            step=0.5,
        )

        stress_level = st.slider(
            "Stress Level (1-10)",
            min_value=1,
            max_value=10,
            value=5,
        )

    with col3:
        family_history_diabetes = st.selectbox(
            "Family History of Diabetes",
            ["No", "Yes"],
        )

        physical_activity_level = st.selectbox(
            "Physical Activity Level",
            ["Sedentary", "Moderate", "Active"],
        )

        diet_type = st.selectbox(
            "Diet Type",
            ["Vegetarian", "Non-Vegetarian", "Pescatarian", "Vegan"],
        )

        smoking_status = st.selectbox(
            "Smoking Status",
            ["Never", "Current", "Former"],
        )

        alcohol_consumption = st.selectbox(
            "Alcohol Consumption",
            ["Never", "Occasional", "Regular"],
        )

    submit_button = st.form_submit_button(
        label="🔍 Evaluate Diabetes Risk",
        use_container_width=True,
    )


# ============================================================
# PDF REPORT GENERATION
# ============================================================

def create_pdf_report(patient_data, predicted_label, pred_proba):
    """Create a PDF clinical assessment report."""

    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(
        0,
        10,
        txt="Clinical Diabetes Risk Assessment Report",
        ln=True,
        align="C",
    )

    pdf.set_font("Arial", size=10)
    pdf.cell(
        0,
        8,
        txt=(
            f"Generated on: "
            f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
        ),
        ln=True,
        align="C",
    )

    pdf.ln(8)

    # Assessment results
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, txt="Assessment Results:", ln=True)

    pdf.set_font("Arial", size=11)
    pdf.cell(
        0,
        6,
        txt=f"Predicted Risk Tier: {predicted_label}",
        ln=True,
    )

    # Safely handle probability output
    class_probabilities = list(pred_proba)

    for index, probability in enumerate(class_probabilities):
        if index == 0:
            label = "Low Risk"
        elif index == 1:
            label = "Moderate Risk"
        elif index == 2:
            label = "High Risk"
        else:
            label = f"Class {index}"

        pdf.cell(
            0,
            6,
            txt=f" - {label} Probability: {probability * 100:.2f}%",
            ln=True,
        )

    pdf.ln(8)

    # Patient profile
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, txt="Patient Profile Summary:", ln=True)

    pdf.set_font("Arial", size=10)

    for key, value in patient_data.items():
        formatted_key = key.replace("_", " ").title()
        pdf.cell(
            0,
            6,
            txt=f"{formatted_key}: {value}",
            ln=True,
        )

    pdf.ln(8)

    # Disclaimer
    pdf.set_font("Arial", "B", 10)
    pdf.cell(0, 6, txt="Medical Disclaimer:", ln=True)

    pdf.set_font("Arial", size=9)

    disclaimer = (
        "This report provides a model-based risk estimate and is not "
        "a medical diagnosis. Results should not replace professional "
        "medical evaluation, laboratory testing, or clinical judgment."
    )

    pdf.multi_cell(0, 5, disclaimer)

    return pdf.output(dest="S").encode("latin1")


# ============================================================
# PREDICTION
# ============================================================

if submit_button:

    input_dict = {
        "age": age,
        "gender": gender,
        "city": city,
        "bmi": bmi,
        "family_history_diabetes": family_history_diabetes,
        "physical_activity_level": physical_activity_level,
        "diet_type": diet_type,
        "smoking_status": smoking_status,
        "alcohol_consumption": alcohol_consumption,
        "hours_sleep_per_night": hours_sleep_per_night,
        "stress_level": stress_level,
        "fasting_blood_sugar": fasting_blood_sugar,
        "hba1c_level": hba1c_level,
        "blood_pressure_systolic": blood_pressure_systolic,
        "blood_pressure_diastolic": blood_pressure_diastolic,
        "waist_circumference_cm": waist_circumference_cm,
        "income_bracket": income_bracket,
    }

    input_data = pd.DataFrame([input_dict])

    try:
        # Model prediction
        pred_class = model.predict(input_data)[0]
        pred_proba = model.predict_proba(input_data)[0]

        # Convert NumPy scalar to normal Python value where appropriate
        try:
            pred_class = int(pred_class)
        except (TypeError, ValueError):
            pass

        risk_mapping = {
            0: "Low Risk",
            1: "Moderate Risk",
            2: "High Risk",
        }

        predicted_label = risk_mapping.get(
            pred_class,
            str(pred_class),
        )

        # ====================================================
        # RESULTS
        # ====================================================

        st.markdown("---")
        st.header("📊 Evaluation Results")

        st.subheader(f"Predicted Risk Tier: {predicted_label}")

        # Probability metrics
        metric_columns = st.columns(len(pred_proba))

        probability_labels = [
            "Low Risk Probability",
            "Moderate Risk Probability",
            "High Risk Probability",
        ]

        for index, probability in enumerate(pred_proba):

            if index < len(probability_labels):
                label = probability_labels[index]
            else:
                label = f"Class {index} Probability"

            metric_columns[index].metric(
                label,
                f"{probability * 100:.2f}%",
            )

        # ====================================================
        # CLINICAL GUIDANCE
        # ====================================================

        if pred_class == 2:

            st.error(
                "⚠️ **High Risk Alert:** "
                "The model estimates a high-risk category. "
                "Consider appropriate clinical follow-up and "
                "confirmatory assessment by a qualified healthcare professional."
            )

        elif pred_class == 1:

            st.warning(
                "⚠️ **Moderate Risk:** "
                "The model estimates a moderate-risk category. "
                "Consider lifestyle assessment, appropriate glucose monitoring, "
                "and professional clinical follow-up."
            )

        else:

            st.success(
                "✔ **Low Risk Category:** "
                "The model estimates a low-risk category. "
                "Continue healthy lifestyle practices and routine preventive care."
            )

        # ====================================================
        # PDF DOWNLOAD
        # ====================================================

        st.markdown("---")

        pdf_bytes = create_pdf_report(
            input_dict,
            predicted_label,
            pred_proba,
        )

        st.download_button(
            label="📄 Download Clinical Assessment Report (PDF)",
            data=pdf_bytes,
            file_name=(
                "Diabetes_Risk_Report_"
                f"{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            ),
            mime="application/pdf",
            use_container_width=True,
        )

    except Exception as exc:
        st.error(
            "The prediction could not be completed. "
            "Please verify that the input fields match the variables "
            "used when the model was trained."
        )
        st.exception(exc)


# ============================================================
# APPLICATION FOOTER
# ============================================================

st.divider()

with st.container(horizontal_alignment="center"):
    st.caption("🩺 Clinical Diabetes Risk Prediction Tool")

    st.caption(
        "A data-driven health risk assessment application designed "
        "to support diabetes risk screening and clinical decision-making."
    )

    st.markdown(
        "**Health Analytics**  •  **Risk Assessment**  •  "
        "**Clinical Decision Support**"
    )

    st.warning(
        "**Medical Disclaimer:** This application provides a model-based risk estimate "
        "and is not a medical diagnosis. The prediction should not replace professional "
        "medical evaluation, laboratory testing, clinical judgment, or consultation "
        "with a qualified healthcare professional. Always interpret results together "
        "with the patient's complete clinical history and appropriate diagnostic assessment."
    )

    st.caption(
        "© 2026 Clinical Diabetes Risk Prediction Tool  |  "
        "Health Sector Data Analytics  |  Version 1.0"
    )
