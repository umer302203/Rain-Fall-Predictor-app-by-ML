import streamlit as st
import pandas as pd
import joblib
import datetime

# --- Function to convert Date to Season (Same as Step 6) ---
def date_to_season(date):
    month = date.month
    if (month == 12) or (month == 1) or (month == 2):
        return 'Summer'
    elif (month == 3) or (month == 4) or (month == 5):
        return 'Autumn'
    elif (month == 6) or (month == 7) or (month == 8):
        return 'Winter'
    elif (month == 9) or (month == 10) or (month == 11):
        return 'Spring'


st.set_page_config(page_title="Rain Prediction App", page_icon="🌧️")

st.title("🌧️ Australian Rainfall Prediction App")
st.write("""
### Predicting Rain in Melbourne Area
This app uses a Machine Learning model (Logistic Regression/Random Forest) trained on Melbourne weather data.
""")


try:
    model = joblib.load('rain_model.pkl')
except:
    st.error("Model file 'rain_model.pkl' not found. Please train the model first.")
    st.stop()

# --- Sidebar for Inputs ---
st.sidebar.header("Enter Weather Conditions")

# 1. Location (Only those used in training)
location = st.sidebar.selectbox("Location", ['Melbourne', 'MelbourneAirport', 'Watsonia'])

# 2. Date (To calculate Season)
date_input = st.sidebar.date_input("Date", datetime.date(2024, 1, 1))
season = date_to_season(date_input)

# 3. Rain Yesterday (Target shift logic Step 4)
rain_yesterday = st.sidebar.selectbox("Did it rain yesterday?", ['No', 'Yes'])

# 4. Important Numerical Inputs
st.sidebar.subheader("Temperature & Rain")
min_temp = st.sidebar.number_input("Min Temp (°C)", value=10.0)
max_temp = st.sidebar.number_input("Max Temp (°C)", value=20.0)
rainfall = st.sidebar.number_input("Rainfall (mm)", value=0.0)

st.sidebar.subheader("Wind & Humidity")
wind_speed_9am = st.sidebar.slider("Wind Speed 9am (km/h)", 0, 100, 20)
wind_speed_3pm = st.sidebar.slider("Wind Speed 3pm (km/h)", 0, 100, 25)
humidity_9am = st.sidebar.slider("Humidity 9am (%)", 0, 100, 50)
humidity_3pm = st.sidebar.slider("Humidity 3pm (%)", 0, 100, 40)
pressure_9am = st.sidebar.number_input("Pressure 9am (hPa)", value=1015.0)

# --- Prediction Logic ---
if st.button("Predict Rain Today"):
    # Create Input DataFrame
    # Note: We need to match ALL columns model expects. 
    # For simplicity, filling missing ones with 0 or defaults. 
    # Ideally, gather all inputs, but here we use reasonable defaults for the rest.
    
    input_data = pd.DataFrame({
        'Location': [location],
        'MinTemp': [min_temp],
        'MaxTemp': [max_temp],
        'Rainfall': [rainfall],
        'Evaporation': [5.0],       # Default value
        'Sunshine': [8.0],          # Default value
        'WindGustDir': ['N'],       # Default
        'WindGustSpeed': [30.0],    # Default
        'WindDir9am': ['N'],        # Default
        'WindDir3pm': ['N'],        # Default
        'WindSpeed9am': [wind_speed_9am],
        'WindSpeed3pm': [wind_speed_3pm],
        'Humidity9am': [humidity_9am],
        'Humidity3pm': [humidity_3pm],
        'Pressure9am': [pressure_9am],
        'Pressure3pm': [pressure_9am - 2], # Approx
        'Cloud9am': [3],            # Default
        'Cloud3pm': [3],            # Default
        'Temp9am': [(min_temp+max_temp)/2],
        'Temp3pm': [max_temp - 2],
        'RainYesterday': [rain_yesterday],
        'Season': [season]
    })
    
    # Prediction
    try:
        prediction = model.predict(input_data)
        pred_prob = model.predict_proba(input_data)
        
        st.subheader("Result:")
        if prediction[0] == 'Yes':
            st.error(f"☔ High Chance of Rain Today! (Confidence: {pred_prob[0][1]*100:.1f}%)")
        else:
            st.success(f"☀️ Likely No Rain Today. (Confidence: {pred_prob[0][0]*100:.1f}%)")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.warning("Please check if all columns in the model match the input data.")