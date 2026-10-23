import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="EduPredict Pro", page_icon="🎓", layout="wide")

@st.cache_resource
def load_artifacts():
    data = joblib.load("models/score_predictor.pkl")
    df = pd.read_csv("data/student_data.csv")
    return data["lr_model"], data["dt_model"], data["metrics"], df

try:
    lr_model, dt_model, metrics, df = load_artifacts()
except FileNotFoundError:
    st.error("Model artifacts not found. Please run python train.py first.")
    st.stop()

st.title("🎓 Student Academic Performance Predictor")
st.caption("Engineered by **Team CGPA Overfitters** | Supervised Regression")

tabs = st.tabs(["🚀 Live Prediction", "📊 Model Evaluation", "🔍 Dataset Explorer"])

with tabs[0]:
    col1, col2 = st.columns([1, 1.2], gap="large")
    with col1:
        st.subheader("Candidate Profile")
        study_hours = st.slider("Weekly Study Hours", 1.0, 12.0, 6.0, 0.5)
        attendance = st.slider("Attendance Rate (%)", 50.0, 100.0, 80.0, 1.0)
        past_score = st.slider("Prior Exam Marks (/100)", 30.0, 100.0, 75.0, 1.0)
        model_choice = st.radio("Model", ["Linear Regression", "Decision Tree Regressor"], horizontal=True)

    with col2:
        st.subheader("Forecasted Results")
        input_df = pd.DataFrame({'Study_Hours': [study_hours], 'Attendance_Pct': [attendance], 'Past_Score': [past_score]})
        pred_lr = float(np.clip(lr_model.predict(input_df)[0], 0, 100))
        pred_dt = float(np.clip(dt_model.predict(input_df)[0], 0, 100))
        active_pred = pred_lr if "Linear" in model_choice else pred_dt
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Score", f"{active_pred:.1f} / 100")
        m2.metric("Linear Reg", f"{pred_lr:.1f}")
        m3.metric("Decision Tree", f"{pred_dt:.1f}")
        st.progress(active_pred / 100)

with tabs[1]:
    st.subheader("Model Validation")
    eval_df = pd.DataFrame({
        "Metric": ["MSE", "RMSE", "R² Score"],
        "Linear Regression": [f"{metrics['lr']['mse']:.3f}", f"{metrics['lr']['rmse']:.3f}", f"{metrics['lr']['r2']:.3f}"],
        "Decision Tree": [f"{metrics['dt']['mse']:.3f}", f"{metrics['dt']['rmse']:.3f}", f"{metrics['dt']['r2']:.3f}"]
    })
    st.table(eval_df)

with tabs[2]:
    st.subheader("Sample Records")
    st.dataframe(df.head(10), use_container_width=True)