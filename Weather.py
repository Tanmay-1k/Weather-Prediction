import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df=pd.read_csv('weather_classification_data.csv')

preprocessor = ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),['Cloud Cover','Season','Location'])], remainder='passthrough')
model = Pipeline([
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])


label_cols=['Weather Type','Season','Location','Cloud Cover']
encoders= {}

for col in label_cols:
    
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # encode strings to numbers
    encoders[col] = le  # store the encoder so we can use it later







y=df['Weather Type']
X = df.drop(['Weather Type'],axis = 1)

model.fit(X,y)







import streamlit as st

st.title("🌤️ Weather Type Predictor")


# Input widgets
temp = st.slider("Temperature (°C)", -20, 50, 25)
humidity = st.slider("Humidity (%)", 0, 100, 50)
wind_speed = st.slider("Wind Speed", 0.0, 30.0, 5.0)
precip = st.slider("Precipitation (%)", 0.0, 120.0, 20.0)
cloud_cover = st.selectbox("Cloud Cover", encoders['Cloud Cover'].classes_)
pressure = st.number_input("Atmospheric Pressure", value=1010.0)
uv_index = st.slider("UV Index", 0, 11, 5)
season = st.selectbox("Season", encoders['Season'].classes_)
visibility = st.slider("Visibility (km)", 0.0, 20.0, 5.0)
location = st.selectbox("Location", encoders['Location'].classes_)

# Encode inputs
input_data = pd.DataFrame({
    'Temperature': [temp],
    'Humidity': [humidity],
    'Wind Speed': [wind_speed],
    'Precipitation (%)': [precip],
    'Cloud Cover': [encoders['Cloud Cover'].transform([cloud_cover])[0]],
    'Atmospheric Pressure': [pressure],
    'UV Index': [uv_index],
    'Season': [encoders['Season'].transform([season])[0]],
    'Visibility (km)': [visibility],
    'Location': [encoders['Location'].transform([location])[0]],
})

# Predict
if st.button("Predict Weather Type"):
    pred = model.predict(input_data)[0]
    result = encoders['Weather Type'].inverse_transform([pred])[0]
    st.success(f"🌈 Predicted Weather Type: **{result}**")


