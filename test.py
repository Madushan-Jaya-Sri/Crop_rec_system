import os
import requests
from geopy.geocoders import Nominatim
import random
from dotenv import load_dotenv

load_dotenv()

def get_current_location():
    """
    Get the current location using multiple IP geolocation services.
    Returns city name, latitude and longitude.
    """
    print("\n=== Debug Point: Fetching Current Location ===")
    try:
        # First attempt: using ipapi.co (most reliable first)
        response = requests.get('https://ipapi.co/json/', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✓ Location found via ipapi.co:")
            print(f"  City: {data.get('city', 'Unknown')}")
            print(f"  Region: {data.get('region', 'Unknown')}")
            print(f"  Country: {data.get('country_name', 'Unknown')}")
            print(f"  Coordinates: ({data.get('latitude')}, {data.get('longitude')})")
            if all(key in data for key in ['city', 'latitude', 'longitude']):
                return data['city'], data['latitude'], data['longitude']

        # Second attempt: using ip-api.com
        print("\nTrying alternative service (ip-api.com)...")
        response = requests.get('http://ip-api.com/json')
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                print("✓ Location found via ip-api.com:")
                print(f"  City: {data.get('city', 'Unknown')}")
                print(f"  Region: {data.get('regionName', 'Unknown')}")
                print(f"  Country: {data.get('country', 'Unknown')}")
                print(f"  Coordinates: ({data.get('lat')}, {data.get('lon')})")
                return data['city'], data['lat'], data['lon']

        # Third attempt: using ipinfo.io
        print("\nTrying final service (ipinfo.io)...")
        response = requests.get('https://ipinfo.io/json')
        if response.status_code == 200:
            data = response.json()
            if 'loc' in data and 'city' in data:
                lat, lon = data['loc'].split(',')
                print("✓ Location found via ipinfo.io:")
                print(f"  City: {data.get('city', 'Unknown')}")
                print(f"  Region: {data.get('region', 'Unknown')}")
                print(f"  Country: {data.get('country', 'Unknown')}")
                print(f"  Coordinates: ({lat}, {lon})")
                return data['city'], float(lat), float(lon)

        raise Exception("Could not determine location using any available service")

    except Exception as e:
        print("✗ Error getting location:")
        print(f"  Error details: {str(e)}")
        return None, None, None

def get_coordinates_from_city(city_name, api_key):
    """
    Get coordinates from city name using Nominatim geocoder.
    """
    print(f"\n=== Debug Point: Getting Coordinates for {city_name} ===")
    try:
        geolocator = Nominatim(user_agent="crop_prediction_app")
        location = geolocator.geocode(city_name)
        if location:
            print("✓ Location found:")
            print(f"  Address: {location.address}")
            print(f"  Coordinates: ({location.latitude}, {location.longitude})")
            return location.latitude, location.longitude
        print("✗ Location not found")
        return None, None
    except Exception as e:
        print("✗ Error getting coordinates:")
        print(f"  Error details: {str(e)}")
        return None, None


import requests
from datetime import datetime, timedelta
import statistics

def get_weather(lat, lon, api_key):
    """
    Get average weather data for the last 12 months using historical endpoint.
    """
    print(f"\n=== Debug Point: Fetching Historical Weather Data ===")
    print(f"Location coordinates: ({lat}, {lon})")

    if not all([lat, lon, api_key]):
        print("✗ Missing required parameters:")
        print(f"  Latitude: {'Present' if lat else 'Missing'}")
        print(f"  Longitude: {'Present' if lon else 'Missing'}")
        print(f"  API Key: {'Present' if api_key else 'Missing'}")
        return None

    try:
        # Calculate dates for last 12 months
        end_date = datetime.now()
        temperatures = []
        humidities = []
        rainfalls = []
        
        # We'll sample one day from each month for the past 12 months
        for month in range(12):
            sample_date = (end_date - timedelta(days=30*month)).strftime('%Y-%m-%d')
            
            url = "http://api.weatherstack.com/historical"
            params = {
                "access_key": api_key,
                "query": f"{lat},{lon}",
                "historical_date": sample_date
            }
            
            print(f"\nFetching data for {sample_date}...")
            response = requests.get(url, params=params)
            print(f"API Response Status: {response.status_code}")
            
            data = response.json()
            if "error" in data:
                print("✗ API Error:")
                print(f"  {data['error'].get('info', 'Unknown error')}")
                continue

            historical_data = data.get('historical', {}).get(sample_date, {})
            if historical_data:
                temperatures.append(historical_data.get('avgtemp', 0))
                humidities.append(historical_data.get('humidity', 0))
                rainfalls.append(historical_data.get('precip', 0))

        # Calculate averages if we have data
        if temperatures or humidities or rainfalls:
            weather_data = {
                'temperature': round(statistics.mean(temperatures) if temperatures else 25, 2),
                'humidity': round(statistics.mean(humidities) if humidities else 75, 2),
                'rainfall': round(statistics.mean(rainfalls) * 30 if rainfalls else 200, 2)  # Multiply by 30 for monthly average
            }
            
            print("\n✓ Historical averages calculated:")
            print(f"  Average Temperature: {weather_data['temperature']}°C")
            print(f"  Average Humidity: {weather_data['humidity']}%")
            print(f"  Average Monthly Rainfall: {weather_data['rainfall']}mm")
            
            return weather_data

        # If no historical data is available, return error message
        print("\n✗ No historical weather data available for the specified location")
        raise ValueError("Unable to retrieve historical weather data. Please check your API key and location coordinates.")

    except Exception as e:
        print("✗ Weather API error:")
        print(f"  Error details: {str(e)}")
        # If API fails, raise an exception with error details
        print("\n✗ Failed to retrieve weather data from the API")
        raise Exception(f"Weather API error: {str(e)}. Please try again later or contact support if the issue persists.")
    
    