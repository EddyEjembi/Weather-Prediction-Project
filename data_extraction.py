import requests
import pandas as pd
import os
from datetime import datetime, timedelta

def get_weather_data(api_key, lat, lon, date_str):
    url = "https://api.weatherapi.com/v1/history.json"
    params = {"key": api_key, "q": f"{lat},{lon}", "dt": date_str}
    response = requests.get(url, params=params, verify=False)
    response.raise_for_status()  # Raise an exception if the request fails
    data = response.json()
    return data

def get_city_coordinates(city):
    city_coords = {
        "Lagos": ("6.5244", "3.3792"),
        "Port Harcourt": ("4.8156", "7.0498"),
        "Kano": ("12.0022", "8.5927"),
        "Abuja": ("9.0579", "7.4951"),
        "Ibadan": ("7.3776", "3.9470"),
        "Ota": ("6.6804", "3.2356")
    }
    return city_coords.get(city)

def main():
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        print("Error: Weather API key not provided.")
        return

    cities = ["Lagos", "Port Harcourt", "Kano", "Abuja", "Ibadan", "Ota"]
    start_date = datetime(2022, 6, 1)
    end_date = datetime(2023, 5, 12)

    city_data = []
    for city in cities:
        lat, lon = get_city_coordinates(city)
        city_data_temp = []
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d %H:%M:%S")
            try:
                data = get_weather_data(api_key, lat, lon, date_str)
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
                city_data_temp.append(weather_features)
            except Exception as e:
                print(f"Error fetching data for {city} on {date_str}: {e}")
            current_date += timedelta(hours=1)
        city_data += city_data_temp

    weather_df = pd.DataFrame(city_data)
    print(weather_df.head(10))

    weather_df.to_csv("weather_data.csv", index=False)

if __name__ == "__main__":
    main()
