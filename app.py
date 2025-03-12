import streamlit as st
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
@st.cache_data
def load_data():
    calories = pd.read_csv("calories.csv")
    exercise = pd.read_csv("exercise.csv")
    
    # Merge datasets
    exercise_df = exercise.merge(calories, on="User_ID")
    exercise_df.drop(columns="User_ID", inplace=True)
    
    # Calculate BMI
    exercise_df["BMI"] = exercise_df["Weight"] / ((exercise_df["Height"] / 100) ** 2)
    exercise_df["BMI"] = round(exercise_df["BMI"], 2)

    # Select features
    exercise_df = exercise_df[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
    exercise_df = pd.get_dummies(exercise_df, drop_first=True)
    
    return exercise_df

# Load dataset
exercise_df = load_data()

# Split data
exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

# Prepare training/testing sets
X_train = exercise_train_data.drop("Calories", axis=1)
y_train = exercise_train_data["Calories"]

X_test = exercise_test_data.drop("Calories", axis=1)
y_test = exercise_test_data["Calories"]

# Train Random Forest model
@st.cache_resource
def train_model():
    model = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
    model.fit(X_train, y_train)
    return model

random_reg = train_model()

# Navigation Menu
st.sidebar.title("Navigation")
screen = st.sidebar.radio("Go to:", ["🏠 Welcome", "📝 User Input & Prediction", "📊 Analysis & Recommendations"])

# **Screen 1: Welcome Page**
if screen == "🏠 Welcome":
    st.title("Welcome to the Personal Fitness Tracker! 🏋️‍♂️")
    st.write("This application predicts the **calories burned** based on your exercise details.")
    st.write("Navigate to 'User Input & Prediction' to enter your details and get a prediction.")
    st.image("fitness.jpg", use_column_width=True)  # Replace with a relevant image

# **Screen 2: User Input & Prediction**
elif screen == "📝 User Input & Prediction":
    st.title("User Input & Prediction 🎯")
    
    # Sidebar User Inputs
    st.sidebar.header("Enter Your Details:")
    age = st.sidebar.slider("Age", 10, 100, 30)
    weight = st.sidebar.slider("Weight (kg)", 30, 150, 70)
    height = st.sidebar.slider("Height (cm)", 120, 220, 170)
    duration = st.sidebar.slider("Duration (min)", 0, 60, 15)
    heart_rate = st.sidebar.slider("Heart Rate", 50, 150, 80)
    body_temp = st.sidebar.slider("Body Temperature (°C)", 35, 42, 37)
    gender_button = st.sidebar.radio("Gender", ("Male", "Female"))

    # Calculate BMI
    bmi = round(weight / ((height / 100) ** 2), 2)
    
    # Encode Gender
    gender = 1 if gender_button == "Male" else 0

    # Create DataFrame
    input_data = pd.DataFrame({
        "Age": [age],
        "BMI": [bmi],
        "Duration": [duration],
        "Heart_Rate": [heart_rate],
        "Body_Temp": [body_temp],
        "Gender_male": [gender]
    })

    st.write("### Your Input Details:")
    st.write(input_data)

    # Align prediction data with model input
    input_data = input_data.reindex(columns=X_train.columns, fill_value=0)

    # Make Prediction
    if st.button("Predict Calories Burned 🔥"):
        with st.spinner("Calculating..."):
            time.sleep(2)  # Simulate loading time
            prediction = random_reg.predict(input_data)
            st.success(f"🔥 You will burn **{round(prediction[0], 2)} kilocalories** during this exercise!")

# **Screen 3: Analysis & Recommendations**
elif screen == "📊 Analysis & Recommendations":
    st.title("Analysis & Personalized Recommendations 📈")

    # Find similar results
    if 'prediction' in locals():  # Ensure prediction exists
        calorie_range = [prediction[0] - 10, prediction[0] + 10]
        similar_data = exercise_df[
            (exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])
        ]
        
        st.write("### 🔍 Similar Past Exercise Records:")
        st.write(similar_data.sample(5))

        st.write("---")
        st.write("### 📊 General Information:")

        boolean_age = (exercise_df["Age"] < input_data["Age"].values[0]).tolist()
        boolean_duration = (exercise_df["Duration"] < input_data["Duration"].values[0]).tolist()
        boolean_body_temp = (exercise_df["Body_Temp"] < input_data["Body_Temp"].values[0]).tolist()
        boolean_heart_rate = (exercise_df["Heart_Rate"] < input_data["Heart_Rate"].values[0]).tolist()

        st.write(f"You are older than **{round(sum(boolean_age) / len(boolean_age), 2) * 100}%** of other users.")
        st.write(f"Your exercise duration is higher than **{round(sum(boolean_duration) / len(boolean_duration), 2) * 100}%** of users.")
        st.write(f"Your heart rate is higher than **{round(sum(boolean_heart_rate) / len(boolean_heart_rate), 2) * 100}%** of users.")
        st.write(f"Your body temperature is higher than **{round(sum(boolean_body_temp) / len(boolean_body_temp), 2) * 100}%** of users.")

        st.write("---")
        st.write("### 💡 Personalized Recommendations:")

        recommendations = []

        # BMI Analysis
        if input_data["BMI"].values[0] < 18.5:
            recommendations.append("🔹 Your BMI is low. Consider adding more protein and calorie-dense foods.")
        elif input_data["BMI"].values[0] > 25:
            recommendations.append("⚠️ Your BMI is high. Try incorporating more cardio and a balanced diet.")

        # Heart Rate
        if input_data["Heart_Rate"].values[0] > 100:
            recommendations.append("🔴 Your heart rate is high. Reduce exercise intensity or consult a doctor.")
        elif input_data["Heart_Rate"].values[0] < 70:
            recommendations.append("🟢 Your heart rate is lower than normal. Increase workout intensity.")

        # Exercise Duration
        if input_data["Duration"].values[0] < 10:
            recommendations.append("🟠 Increase workout duration to at least 30 minutes per session.")

        # Body Temperature
        if input_data["Body_Temp"].values[0] > 39:
            recommendations.append("⚠️ High body temperature detected. Stay hydrated and avoid overheating.")

        for rec in recommendations:
            st.write(rec)

    else:
        st.write("⚠️ **Please make a prediction first in the User Input screen!**")

