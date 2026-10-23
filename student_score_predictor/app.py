import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="EduPredict Pro", page_icon="🎓", layout="wide")

# ==========================================
# Auto-Train Function if Files Missing on Cloud
# ==========================================
@st.cache_resource
def load_or_train_artifacts():
    model_path = "models/score_predictor.pkl"
    data_path = "data/student_data.csv"
    
    # Agar files locally ya cloud par pehle se maujood hain
    if os.path.exists(model_path) and os.path.exists(data_path):
        payload = joblib.load(model_path)
        df = pd.read_csv(data_path)
        return payload["lr_model"], payload["dt_model"], payload["metrics"], df

    # Agar files nahi milti hain, toh automatically live train karo
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    np.random.seed(42)
    n_samples = 500

    study_hours = np.random.uniform(1.0, 10.0, n_samples)
    attendance_pct = np.random.uniform(50.0, 100.0, n_samples)
    past_score = np.random.uniform(35.0, 95.0, n_samples)

    noise = np.random.normal(0, 3.2, n_samples)
    final_score = (
        (study_hours * 3.65) + 
        (attendance_pct * 0.22) + 
        (past_score * 0.44) + 
        noise
    )
    final_score = np.clip(final_score, 0, 100)

    df = pd.DataFrame({
        'Study_Hours': np.round(study_hours, 1),
        'Attendance_Pct': np.round(attendance_pct, 1),
        'Past_Score': np.round(past_score, 1),
        'Final_Score': np.round(final_score, 1)
    })
    df.to_csv(data_path, index=False)

    X = df[['Study_Hours', 'Attendance_Pct', 'Past_Score']]
    y = df['Final_Score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)

    dt = DecisionTreeRegressor(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    dt_preds = dt.predict(X_test)

    payload = {
        "lr_model": lr,
        "dt_model": dt,
        "metrics": {
            "lr": {
                "mse": mean_squared_error(y_test, lr_preds),
                "rmse": np.sqrt(mean_squared_error(y_test, lr_preds)),
                "r2": r2_score(y_test, lr_preds)
            },
            "dt": {
                "mse": mean_squared_error(y_test, dt_preds),
                "rmse": np.sqrt(mean_squared_error(y_test, dt_preds)),
                "r2": r2_score(y_test, dt_preds)
            }
        }
    }
    joblib.dump(payload, model_path)
    return payload["lr_model"], payload["dt_model"], payload["metrics"], df

lr_model, dt_model, metrics, df = load_or_train_artifacts()

# ==========================================
# Streamlit Dashboard UI
# ==========================================
st.title("🎓 Student Academic Performance Predictor")
st.caption("Engineered by **Team CGPA Overfitters** | Comparative Supervised Regression Suite")

tabs = st.tabs(["🚀 Live Prediction & Simulator", "📊 Model Evaluation & Benchmarks", "🔍 Dataset Explorer"])

# Tab 1: Live Prediction
with tabs[0]:
    col1, col2 = st.columns([1, 1.2], gap="large")
    with col1:
        st.subheader("Candidate Profile")
        study_hours = st.slider("Weekly Dedicated Study Hours", 1.0, 12.0, 6.0, 0.5)
        attendance = st.slider("Semester Attendance Rate (%)", 50.0, 100.0, 80.0, 1.0)
        past_score = st.slider("Previous Examination Grade (/100)", 30.0, 100.0, 75.0, 1.0)
        model_choice = st.radio("Primary Inference Model", ["Linear Regression", "Decision Tree Regressor"], horizontal=True)

    with col2:
        st.subheader("Forecasted Results")
        input_df = pd.DataFrame({'Study_Hours': [study_hours], 'Attendance_Pct': [attendance], 'Past_Score': [past_score]})
        pred_lr = float(np.clip(lr_model.predict(input_df)[0], 0, 100))
        pred_dt = float(np.clip(dt_model.predict(input_df)[0], 0, 100))
        active_pred = pred_lr if "Linear" in model_choice else pred_dt

        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Score", f"{active_pred:.1f} / 100")
        m2.metric("Linear Reg", f"{pred_lr:.1f}")
        m3.metric("Decision Tree", f"{pred_dt:.1f}", delta=f"{pred_dt - pred_lr:.1f}")

        st.progress(active_pred / 100)

        if active_pred >= 80:
            st.success("🎯 **Tier 1 (High Distinction):** Candidate demonstrates strong conceptual retention.")
        elif active_pred >= 60:
            st.info("📈 **Tier 2 (Stable / Passing):** Candidate is on track. Incremental effort will push this to distinction.")
        else:
            st.warning("⚠️ **Tier 3 (Academic Intervention Required):** Projected below passing threshold.")

# Tab 2: Evaluation
with tabs[1]:
    st.subheader("Model Validation & Error Analysis")
    eval_df = pd.DataFrame({
        "Metric": ["Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)", "R² Score (Variance Explained)"],
        "Linear Regression": [f"{metrics['lr']['mse']:.3f}", f"{metrics['lr']['rmse']:.3f}", f"{metrics['lr']['r2']:.3f}"],
        "Decision Tree": [f"{metrics['dt']['mse']:.3f}", f"{metrics['dt']['rmse']:.3f}", f"{metrics['dt']['r2']:.3f}"]
    })
    st.table(eval_df)

# Tab 3: Dataset Explorer
with tabs[2]:
    st.subheader("Dataset Records & Distribution")
    st.dataframe(df.head(15), use_container_width=True)