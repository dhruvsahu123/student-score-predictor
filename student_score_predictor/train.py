import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

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

df.to_csv("data/student_data.csv", index=False)

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

joblib.dump(payload, "models/score_predictor.pkl")
print("Model trained and saved to models/score_predictor.pkl")