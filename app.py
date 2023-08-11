#Import relevant libraries
import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
import pickle
from datetime import datetime, timedelta

# API key for weatherapi.com
API_KEY = 'fb222dc231b44f05b0682536230205'


# Define city coordinates
city_coordinates = {
    "Lagos": ("6.5244", "3.3792"),
    "Port Harcourt": ("4.8156", "7.0498"),
    "Kano": ("12.0022", "8.5927"),
    "Abuja": ("9.0579", "7.4951"),
    "Ibadan": ("7.3776", "3.9470"),
    "Ota": ("6.6804", "3.2356"),
}


#Days to forcast
def get_next_days():
    today = datetime.today().date()
    dat = pd.date_range(today, periods=8).date
    return dat

#load machine learning model
with open('ensemble_model.pkl', 'rb') as file:
    ensemble = pickle.load(file)

# Function to fetch hourly weather data for a given date and city
@st.cache_data
def fetch_hourly_weather_data(date, city, lat, lon, api_key):
    # API endpoint for weather data
    url = f"https://api.exampleweatherapi.com/v1/hourly?lat={lat}&lon={lon}&date={date}&key={api_key}"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception if the request was not successful
        data = response.json()

        # Extract relevant hourly weather data
        hourly_data = data.get("hourly", [])

        # List to store weather data
        weather_data = []
        for hour in hourly_data:
            # Extract relevant features from API response
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
            # Append features to weather data list
            weather_data.append(weather)

        return weather_data

    except requests.exceptions.RequestException as e:
        print("Error fetching weather data:", e)
        return None

# Function to perform section-based prediction
def predict_section_weather_conditions(section_data, model, city_encode):
    # Drop the columns not needed in prediction
    features = [
        "Atmospheric Pressure", "Cloud Cover (%)", "Dew Point", "UV Index",
        "Visibility", "Wind Gust", "humidity", "precip_mm", "temp_c", "wind_kmph",
        "hour", "hour_sin", "hour_cos"
    ]
    x = section_data[features].drop(['condition', 'city', 'date_time', 'year', 'Section of Day'], axis=1)

    # Make prediction
    predictions = model.predict(x)

    # Decode the predicted condition labels
    decoded_condition = {
        0: 'Clear', 1: 'Cloudy', 2: 'Heavy Rain at Times', 3: 'Light Rain Shower',
        4: 'Mist', 5: 'Moderate or Heavy Rain Shower', 6: 'Moderate Rain at Times',
        7: 'Overcast', 8: 'Partly Cloudy', 9: 'Patchy Light Rain with Thunder',
        10: 'Patchy Rain Possible', 11: 'Thundry Outbreak Possible'
    }
    predicted_conditions = [decoded_condition[label] for label in predictions]

    # Create a DataFrame with the predictions and the city encoding
    result_df = section_data.copy()
    result_df['Predicted Condition'] = predicted_conditions
    result_df['city_encoded'] = city_encode

    return result_df


# Function to decode weather conditions
def decode_weather_condition(prediction, decoded_condition):
    # Check if the prediction label exists in the decoded_condition dictionary
    if prediction in decoded_condition:
        return decoded_condition[prediction]
    else:
        return "Unknown"


#Main app function
def main():
    st.title("Weather Prediction App")

    col1, col2 = st.columns(2)

    # User inputs
    cities = ["Lagos", "Port Harcourt", "Kano", "Abuja", "Ibadan", "Ota"]

    # Fetch the next 3 days
    dat = get_next_days()

    #Input fields
    with col1:
        city = st.selectbox("Select the city:", cities)
    with col2:
        dates = st.multiselect("Select the date(s) (YYYY-MM-DD):", dat)

    # Fetch the next 3 days
    if not dates:
        st.warning("Please select at least one date.")
        return

    lat, lon = city_coordinates.get(city, ("0", "0"))
    city_encode = cities.index(city)

    for date in dates:
        print(date)
        weather_data = fetch_hourly_weather_data(date, city, lat, lon)

        # If data is fetched
        if weather_data:
            # Process weather data (similar to your code, but moved to a separate function)

            # Perform section-based prediction
            predictions, section_result = predict_section_weather_conditions(df, ensemble, city_encode)

            # Decode weather conditions
            decoded_conditions = [decode_weather_condition(pred, decoded_condition) for pred in predictions]

            # Display predictions (similar to your code, but refactored)
        else:
            st.error("Failed to fetch weather data. Please check your inputs.")


    #Cities to predict for
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
        
        # Fetch the hourly weather data for dates selected
        for date in dates:
            print(date)
            weather_data = fetch_hourly_weather_data(date, city, lat, lon)

            #If data is fetched
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

                # Divide the Day into 4 Sections
                section_data = df
                section_data['Section of Day'] = pd.cut(df['hour'], bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'])
                #print(section_data)

                #Create 4 different dataframe for each section
                night = section_data[section_data['Section of Day'] == 'Night']
                morning = section_data[section_data['Section of Day'] == 'Morning']
                afternoon = section_data[section_data['Section of Day'] == 'Afternoon']
                evening = section_data[section_data['Section of Day'] == 'Evening']
                #print(night)
                #Compute the aggregate for each section
                features_2 = ['hour', 'hour_sin', 'hour_cos']
                night[features + features_2] = night[features + features_2].mean()
                morning[features + features_2] = morning[features +features_2].mean()
                afternoon[features + features_2] = afternoon[features + features_2].mean()
                evening[features + features_2] = evening[features + features_2].mean()
                #print(night)
                #variable to hold sections data
                section = [night, morning, afternoon, evening]
                #List to store result of section prediction
                section_result = []

                #prediction for section data
                for i in section:
                    #Drop the columns not needed in prediction
                    x = i.drop(['condition', 'city', 'date_time', 'year', 'Section of Day'], axis=1)
                    #Make prediction
                    pred = ensemble.predict(x)
                    #Get the variable name of the section and store it as a string
                    var_name = [name for name, value in locals().items() if value is i][0]
                    result = [var_name.capitalize(), pred.mean()]
                    #Append the section and prediction to the section result List
                    section_result.append(result)
                print(section_result)

                #prdiction for hourly data
                #Drop the columns not needed in prediction
                data = df.drop(['condition', 'city', 'date_time', 'year', 'Section of Day'], axis=1)
                #Make prediction for every hour of the day
                predictions = ensemble.predict(data)

                #Dictionary to decode the Weather Conditions
                decoded_condition = {
                    'Clear': [0, ':sun_with_face:'],
                    'Cloudy': [1, ':cloud:'],
                    'Heavy Rain at Times': [2, ':rain_cloud:'],
                    'Light Rain Shower': [3, ':partly_sunny_rain:'],
                    'Mist': [4, ':fog:'],
                    'Moderate or Heavy Rain Shower': [5, ':rain_cloud:'],
                    'Moderate Rain at Times': [6, ':umbrella:'],
                    'Overcast': [7, ':cloud:'],
                    'Partly Cloudy': [8, ':barely_sunny:'],
                    'Patchy Light Rain with Thunder': [9, ':thunder_cloud_and_rain:'],
                    'Patchy Rain Possible': [10, ':partly_sunny_rain:'],
                    'Thundry Outbreak Possible': [11, ':lightning:'],
                }

                # Create a DataFrame with the predictions
                section_data['condition'] = pd.DataFrame(df['condition'])
                section_data['Predicted Condition Encoded'] = pd.DataFrame(predictions)

                pred = None
                emoji = None
                #List to hold the decoded Predictions for every hour
                decode = []

                #Decode the condition and append it to the List
                for i in predictions:
                    for key, value in decoded_condition.items():
                        if value[0] == i:
                            decode.append(key)

                # Print the predicted conditions for each section
                st.write("")
                st.write("")
                st.header("Predicted Weather Condition for {}:".format(date))
                #st.write('At {} (00:00 - 05:00) expect a \"{}\" Weather'.format(dis, pred.upper()))
                #Loop through the section result and display the section along with it's prediction
                for i in section_result:
                    for key, value in decoded_condition.items():
                        if value[0] == i[1]:
                            pred = key
                            emoji = value[1]
                            break
                    if i[0] == 'Night':
                        st.write('At {} :night_with_stars: (00:00 - 05:00) expect a \"{} {}\"  Weather'.format(i[0], pred.upper(), emoji))
                    if i[0] == 'Morning':
                        st.write('In the {} :city_sunrise: (06:00 - 11:00) expect a \"{} {}\"  Weather'.format(i[0], pred.upper(), emoji))
                    if i[0] == 'Afternoon':
                        st.write('In the {} :sunrise: (12:00 - 17:00) expect a \"{} {}\"  Weather'.format(i[0], pred.upper(), emoji))
                    if i[0] == 'Evening':
                        st.write('In the {} :city_sunset: (18:00 - 23:00) expect a \"{} {}\"  Weather'.format(i[0], pred.upper(), emoji))
                
                #Display the predictions
                section_data['Predicted Condition'] = decode

                st.subheader("Predicted Weather Conditions Every hour:")
                columns_to_not_display = ['condition', 'city', 'date_time', 'city_encoded', 'year', 'month', 'day', 'hour',
                                         'month_sin', 'month_cos', 'hour_sin', 'hour_cos', 'Predicted Condition Encoded']
                st.dataframe(section_data.drop(columns_to_not_display, axis=1))
            else:
                st.error("Failed to fetch weather data. Please check your inputs.")
    else:
        st.warning("Please enter a city and date.")
        
    st.markdown("---")
    #st.markdown("© 2023 [Eddy Ejembi](https://github.com/EddyEjembi). All rights reserved.")

if __name__ == "__main__":
    main()

