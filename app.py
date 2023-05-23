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

#load machine learning model
with open('ensemble_model.pkl', 'rb') as file:
    ensemble = pickle.load(file)

# Function to fetch hourly weather data for a given date and city
@st.cache_data
def fetch_hourly_weather_data(date, city, lat, lon):
    date = date.strftime('%Y-%m-%d')  # Convert date to string
    #make API request
    url = 'http://api.weatherapi.com/v1/forecast.json?key=' + API_KEY + '&q=' + str(lat) + ',' + str(lon) + '&dt=' + date + '&hour=0-23'
    response = requests.get(url)
    data = response.json()
    #print(response)
    #Parameter for API request
    hourly_data = data.get('forecast', {}).get('forecastday', [])[0].get('hour', [])

    #List to store weather data
    weather_data = []
    for hour in hourly_data:
        #extract relevant features from API
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
        #Append features to weather data List
        weather_data.append(weather)

    #Return weather data List
    return weather_data


#Main app function
def main():
    st.title("Weather Prediction App")

    col1, col2 = st.columns(2)

    # User inputs
    cities = ["Lagos", "Port Harcourt", "Kano", "Abuja", "Ibadan", "Ota"]
    #Days to forcast
    def get_next_days():
        today = datetime.today().date()
        dat = pd.date_range(today, periods=4).date
        return dat

    # Fetch the next 3 days
    dat = get_next_days()

    #Input fields
    with col1:
        city = st.selectbox("Select the city:", cities)
    with col2:
        dates = st.multiselect("Select the date(s) (YYYY-MM-DD):", dat)
    lat, lon = 0, 0 #latitude and longitude of the city
    city_encode = 0 #encoded value of the city

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
                section_data['condition'] = pd.DataFrame(df['condition'])
                section_data['Predicted Condition Encoded'] = pd.DataFrame(predictions)

                pred = None
                #List to hold the decoded Predictions for every hour
                decode = []

                #Decode the condition and append it to the List
                for i in predictions:
                    for key, value in decoded_condition.items():
                        if value == i:
                            decode.append(key)

                # Print the predicted conditions for each section
                st.write("")
                st.write("")
                st.header("Predicted Weather Condition for {}:".format(date))
                #st.write('At {} (00:00 - 05:00) expect a \"{}\" Weather'.format(dis, pred.upper()))
                #Loop through the section result and display the section along with it's prediction
                for i in section_result:
                    for key, value in decoded_condition.items():
                        if value == i[1]:
                            pred = key
                            break
                    if i[0] == 'Night':
                        st.write('At {} (00:00 - 05:00) expect a \"{}\" Weather'.format(i[0], pred.upper()))
                    if i[0] == 'Morning':
                        st.write('In the {} (06:00 - 11:00) expect a \"{}\" Weather'.format(i[0], pred.upper()))
                    if i[0] == 'Afternoon':
                        st.write('In the {} (12:00 - 17:00) expect a \"{}\" Weather'.format(i[0], pred.upper()))
                    if i[0] == 'Evening':
                        st.write('In the {} (18:00 - 23:00) expect a \"{}\" Weather'.format(i[0], pred.upper()))
                
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
    st.markdown("© 2023 [Eddy Ejembi](https://github.com/EddyEjembi). All rights reserved.")

if __name__ == "__main__":
    main()

