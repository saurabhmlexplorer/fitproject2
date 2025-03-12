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

            # Store prediction and input data in session state
            st.session_state["prediction"] = prediction[0]
            st.session_state["user_data"] = input_data

            st.success(f"🔥 You will burn **{round(prediction[0], 2)} kilocalories** during this exercise!")

# **Screen 3: Analysis & Recommendations**
elif screen == "📊 Analysis & Recommendations":
    st.title("Analysis & Personalized Recommendations 📈")

    # Ensure we have a prediction stored
    if "prediction" in st.session_state and "user_data" in st.session_state:
        prediction = st.session_state["prediction"]
        input_data = st.session_state["user_data"].copy()

        
        
        # Find similar results
        calorie_range = [prediction - 10, prediction + 10]
        similar_data = exercise_df[
            (exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])
        ]

        # Convert 'Gender_male' back to categorical values
        similar_data_display = similar_data.copy()
        similar_data_display["Gender"] = similar_data_display["Gender_male"].apply(lambda x: "Male" if x == 1 else "Female")
        similar_data_display.drop(columns=["Gender_male"], inplace=True)  # Remove the encoded column
        
        st.write("### 🔍 Similar Exercise Records:")
        st.write(similar_data_display.sample(5))

        st.write("---")
        st.write("### 📊 General Information:")

        boolean_age = (exercise_df["Age"] < input_data["Age"].values[0]).tolist()
        boolean_duration = (exercise_df["Duration"] < input_data["Duration"].values[0]).tolist()
        boolean_body_temp = (exercise_df["Body_Temp"] < input_data["Body_Temp"].values[0]).tolist()
        boolean_heart_rate = (exercise_df["Heart_Rate"] < input_data["Heart_Rate"].values[0]).tolist()

        st.write(f"You are older than **{round((sum(boolean_age) / len(boolean_age)) * 100, 2)}%** of other users.")
        st.write(f"Your exercise duration is higher than **{round((sum(boolean_duration) / len(boolean_duration)) * 100, 2)}%** of users.")
        st.write(f"Your heart rate is higher than **{round((sum(boolean_heart_rate) / len(boolean_heart_rate)) * 100, 2)}%** of users.")
        st.write(f"Your body temperature is higher than **{round((sum(boolean_body_temp) / len(boolean_body_temp)) * 100, 2)}%** of users.")

        st.write("---")
        st.write("### 💡 Personalized Recommendations:")

        recommendations = []

        # Retrieve values safely
        bmi_value = input_data["BMI"].values[0]
        heart_rate_value = input_data["Heart_Rate"].values[0]
        duration_value = input_data["Duration"].values[0]
        body_temp_value = input_data["Body_Temp"].values[0]

        # BMI Analysis
        if bmi_value < 18.5:
            recommendations.append("🔹 Your BMI is low. Consider adding more protein and calorie-dense foods.")
        elif bmi_value > 25:
            recommendations.append("⚠️ Your BMI is high. Try incorporating more cardio and a balanced diet.")

        # Heart Rate
        if heart_rate_value > 100:
            recommendations.append("🔴 Your heart rate is high. Reduce exercise intensity or consult a doctor.")
        elif heart_rate_value < 70:
            recommendations.append("🟢 Your heart rate is lower than normal. Increase workout intensity.")

        # Exercise Duration
        if duration_value < 10:
            recommendations.append("🟠 Increase workout duration to at least 30 minutes per session.")

        # Body Temperature
        if body_temp_value > 39:
            recommendations.append("⚠️ High body temperature detected. Stay hydrated and avoid overheating.")

        # Display recommendations
        if recommendations:
            for rec in recommendations:
                st.write(rec)
        else:
            st.write("✅ You are within a healthy range. Keep up the good work!")

    else:
        st.write("⚠️ **Please make a prediction first in the User Input screen!**")
