import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

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
    df[col] = le.fit_transform(df[col])  
    encoders[col] = le  







y=df['Weather Type']
X = df.drop(['Weather Type'],axis = 1)

model.fit(X,y)







import streamlit as st

page = st.sidebar.selectbox("Go to", ["Home", "Predict", "Performance"])

if page == "Home":
    st.title("🌦️ Weather Predictor")
    st.write("This Streamlit-based web application predicts the weather type (e.g., Sunny, Rainy, Cloudy, Snowy) using input features like temperature, humidity, wind speed, and more. It supports multipage navigation, allowing users to seamlessly switch between the home screen, prediction interface, and an about section. Powered by machine learning, the app provides quick and interactive forecasts in a clean, user-friendly interface.")
    st.markdown('[The dataset used :],https://www.kaggle.com/datasets/nikhil7280/weather-type-classification')

elif page == "Predict":
    st.title("🔮 Make a Prediction")
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




elif page == "Performance":
    st.title("Model Performance")

    # Assuming you have pipe, X_test, y_test
    y_pred = model.predict(X)

    st.subheader("Classification Report")
    report = classification_report(y, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.dataframe(df_report)

    st.subheader("Accuracy Score")
    acc = accuracy_score(y, y_pred)
    st.metric("Accuracy", f"{acc:.2f}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

   
