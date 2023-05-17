import requests
import pandas as pd

# WeatherAPI.com API endpoint for historical weather data
url = "https://api.weatherapi.com/v1/history.json"

# Your WeatherAPI.com API key
api_key = "<API_KEY>"

# List of cities to obtain weather data for
cities = ["Lagos", "Port Harcourt", "Kano", "Abuja", "Ibadan", "Ota"]

# Date range for which to obtain historical weather data
start_date = "2022-06-01"
end_date = "2023-05-12"

# List to store weather data for each city
city_data = []

# Loop through cities and obtain weather data for each city and date
for city in cities:
    # Latitude and longitude for the city
    if city == "Lagos":
        lat, lon = "6.5244", "3.3792"
    elif city == "Port Harcourt":
        lat, lon = "4.8156", "7.0498"
    elif city == "Kano":
        lat, lon = "12.0022", "8.5927"
    elif city == "Abuja":
        lat, lon = "9.0579", "7.4951"
    elif city == "Ibadan":
        lat, lon = "7.3776", "3.9470"
    elif city == "Ota":
        lat, lon = "6.6804", "3.2356"

    # List to store weather data for the city
    city_data_temp = []
    
    # Loop through dates and hours and obtain weather data for each hour
    for date in pd.date_range(start=start_date, end=end_date, freq="H"):
        # Format date as string
        date_str = date.strftime("%Y-%m-%d %H:%M:%S")
        
        # Parameters for API request
        params = {"key":api_key, "q": lat + "," + lon, "dt":date_str}
        
        # Make API request
        response = requests.get(url, params=params, verify=False)
        
        # Parse JSON data
        data = response.json()
        
        # Extract relevant weather features from data
        weather_features = {
            "city": city,
            "date": date_str,
            "temp_c": data["forecast"]["forecastday"][0]["hour"][0]["temp_c"],
            "humidity": data["forecast"]["forecastday"][0]["hour"][0]["humidity"],
            "wind_kmph": data["forecast"]["forecastday"][0]["hour"][0]["wind_kph"],
            "precip_mm": data["forecast"]["forecastday"][0]["hour"][0]["precip_mm"],
            "Atmospheric Pressure": data["forecast"]["forecastday"][0]["hour"][0]["pressure_mb"],
            "Visibility": data["forecast"]["forecastday"][0]["hour"][0]["vis_km"],
            "Dew Point": data["forecast"]["forecastday"][0]["hour"][0]["dewpoint_c"],
            "Wind Gust": data["forecast"]["forecastday"][0]["hour"][0]["wind_kph"],
            "Cloud Cover (%)": data["forecast"]["forecastday"][0]["hour"][0]["cloud"],
            "UV Index": data["forecast"]["forecastday"][0]["day"].get("uv", "N/A"),
            "condition": data["forecast"]["forecastday"][0]["hour"][0]["condition"]["text"]
        }

        # Append weather features to city data list
        city_data_temp.append(weather_features)
    
    # Append city data to overall city data list
    city_data += city_data_temp

# Convert city data list to Pandas DataFrame
weather_df = pd.DataFrame(city_data)
print(weather_df.head(10))

# Save DataFrame to CSV file
weather_df.to_csv("weather_data.csv", index=False)