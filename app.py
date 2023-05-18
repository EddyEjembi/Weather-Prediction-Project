import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
import pickle
from datetime import datetime, timedelta

# API key for weatherapi.com
API_KEY = 'a62c65107ba54eab825202117231205'

#load machine learning models
with open('ensemble_model.pkl', 'rb') as file:
    ensemble = pickle.load(file)

# Function to fetch hourly weather data for a given date and city
@st.cache_data
def fetch_hourly_weather_data(date, city, lat, lon):
    date = date.strftime('%Y-%m-%d')  # Convert date to string
    url = 'http://api.weatherapi.com/v1/forecast.json?key=' + API_KEY + '&q=' + str(lat) + ',' + str(lon) + '&dt=' + date + '&hour=0-23'
    response = requests.get(url)
    data = response.json()
    #print(response)
    hourly_data = data.get('forecast', {}).get('forecastday', [])[0].get('hour', [])

    weather_data = []
    for hour in hourly_data:
        weather = {
            "city": city,
            "date_time": pd.to_datetime(hour["time"]),
            "temp_c": hour["temp_c"],
            "humidity": hour["humidity"],
            "wind_kmph": hour["wind_kph"],
            "precip_mm": hour["precip_mm"],
            "Atmospheric Pressure": hour["pressure_mb"],
            "Visibility": hour["vis_km"],
            "Dew Point": hour["dewpoint_c"],
            "Wind Gust": hour["wind_kph"],
            "Cloud Cover (%)": hour["cloud"],
            "UV Index": hour.get("uv", "N/A"),
            "condition": hour["condition"]["text"]
        }
        weather_data.append(weather)

    return weather_data


def main():
    st.title("Weather Prediction App")

    col1, col2 = st.columns(2)

    # User inputs
    cities = ["Lagos", "Port Harcourt", "Kano", "Abuja", "Ibadan", "Ota"]
    def get_next_days():
        today = datetime.today().date()
        dat = pd.date_range(today, periods=7).date
        return dat

    # Fetch the next 7 days
    dat = get_next_days()

    with col1:
        city = st.selectbox("Select the city:", cities)
    with col2:
        dates = st.multiselect("Select the date(s) (YYYY-MM-DD):", dat)
    lat, lon = 0, 0 #latitude and longitude of the city
    city_encode = 0 #encoded value of the city

    if city and dates:
        if city == "Lagos":
            lat, lon = "6.5244", "3.3792"
            city_encode = 3
        elif city == "Port Harcourt":
            lat, lon = "4.8156", "7.0498"
            city_encode = 5
        elif city == "Kano":
            lat, lon = "12.0022", "8.5927"
            city_encode = 2
        elif city == "Abuja":
            lat, lon = "9.0579", "7.4951"
            city_encode = 0
        elif city == "Ibadan":
            lat, lon = "7.3776", "3.9470"
            city_encode = 1
        elif city == "Ota":
            lat, lon = "6.6804", "3.2356"
            city_encode = 4
        # Fetch the hourly weather data
        for date in dates:
            print(date)
            weather_data = fetch_hourly_weather_data(date, city, lat, lon)

            if weather_data:
                # Convert the weather data to a pandas DataFrame
                df = pd.DataFrame(weather_data)
                df = df.sort_index(axis=1)
                df['city_encoded'] = city_encode

                # Perform date and time processing
                df['year'] = df['date_time'].dt.year
                df['month'] = df['date_time'].dt.month
                df['day'] = df['date_time'].dt.day
                df['hour'] = df['date_time'].dt.hour

                # Perform cyclical encoding on month and hour features
                df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
                df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
                df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
                df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

                # Select the relevant features for prediction
                features = ["Atmospheric Pressure", "Cloud Cover (%)", "Dew Point", "UV Index",
                            "Visibility", "Wind Gust", "humidity", "precip_mm", "temp_c", "wind_kmph"]

                # Scale the features using StandardScaler
                scaler = StandardScaler()
                # Fit the scaler on the selected features
                scaler.fit(df[features])
                # Transform the selected features using the scaler
                df[features] = scaler.transform(df[features])

                # Make predictions using the ensemble model
                data = df.drop(['condition', 'city', 'date_time', 'year'], axis=1)
                predictions = ensemble.predict(data)

                decoded_condition = {
                    'Clear': 0,
                    'Cloudy': 1,
                    'Heavy Rain at Times': 2,
                    'Light Rain Shower': 3,
                    'Mist': 4,
                    'Moderate or Heavy Rain Shower': 5,
                    'Moderate Rain at Times': 6,
                    'Overcast': 7,
                    'Partly Cloudy': 8,
                    'Patchy Light Rain with Thunder': 9,
                    'Patchy Rain Possible': 10,
                    'Thundry Outbreak Possible': 11

                }

                # Create a DataFrame with the predictions
                result = pd.DataFrame(df['condition'])
                result['Predicted Condition Encoded'] = pd.DataFrame(predictions)

                # Divide the day into sections (e.g., 3 sections: morning, afternoon, evening)
                result['Section of Day'] = pd.cut(df['hour'], bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'])

                # Group the data by sections and calculate the average predictions
                section_predictions = result.groupby('Section of Day')['Predicted Condition Encoded'].mean()
                pred = None
                decode = []

                for i in predictions:
                    for key, value in decoded_condition.items():
                        if value == i:
                            decode.append(key)

                # Print the predicted conditions for each section
                st.write("")
                st.write("")
                st.header("Predicted Weather Condition for {}:".format(date))
                for section, prediction in section_predictions.items():
                    print(f'{section}: {prediction}')
                    for key, value in decoded_condition.items():
                        if value == np.round(prediction):
                            pred = key
                            break
                    if section == 'Night':
                        st.write('At {} (00:00 - 05:00) expect a \"{}\" Weather'.format(section, pred.upper()))
                    if section == 'Morning':
                        st.write('In the {} (06:00 - 11:00) expect a \"{}\" Weather'.format(section, pred.upper()))
                    if section == 'Afternoon':
                        st.write('In the {} (12:00 - 17:00) expect a \"{}\" Weather'.format(section, pred.upper()))
                    if section == 'Evening':
                        st.write('In the {} (18:00 - 23:00) expect a \"{}\" Weather'.format(section, pred.upper()))
                    #st.write(f'{section}: {prediction}')


                result['Predicted Condition'] = decode
                # Display the predictions
                st.subheader("Predicted Weather Conditions Every hour:")
                st.dataframe(result.drop(['condition'], axis=1))
            else:
                st.error("Failed to fetch weather data. Please check your inputs.")
    else:
        st.warning("Please enter a city and date.")
        
    st.markdown("---")
    st.markdown("© 2023 [Eddy Ejembi](https://github.com/EddyEjembi). All rights reserved.")

if __name__ == "__main__":
    main()

