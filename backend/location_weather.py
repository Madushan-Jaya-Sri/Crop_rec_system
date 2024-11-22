import os
import requests
from geopy.geocoders import Nominatim
import random

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

def get_weather(lat, lon, api_key):
    """
    Get weather data from API without soil type adjustments.
    """
    print(f"\n=== Debug Point: Fetching Weather Data ===")
    print(f"Location coordinates: ({lat}, {lon})")

    if not all([lat, lon, api_key]):
        print("✗ Missing required parameters:")
        print(f"  Latitude: {'Present' if lat else 'Missing'}")
        print(f"  Longitude: {'Present' if lon else 'Missing'}")
        print(f"  API Key: {'Present' if api_key else 'Missing'}")
        return None

    try:
        url = "http://api.weatherstack.com/current"
        params = {
            "access_key": api_key,
            "query": f"{lat},{lon}"
        }
        
        print("\nMaking API request to Weatherstack...")
        response = requests.get(url, params=params)
        print(f"API Response Status: {response.status_code}")
        
        data = response.json()
        if "error" in data:
            print("✗ API Error:")
            print(f"  {data['error'].get('info', 'Unknown error')}")
            return None

        current_data = data.get('current', {})
        location_data = data.get('location', {})
        
        print("\n✓ Weather data received:")
        print(f"  Location: {location_data.get('name', 'Unknown')}, {location_data.get('country', 'Unknown')}")
        
        # Get actual weather values from API
        weather_data = {
            'temperature': current_data.get('temperature', 25),
            'humidity': current_data.get('humidity', 75),
            'rainfall': current_data.get('precip', 0) * 30  # Convert daily to monthly
        }

        print("\nWeather values:")
        print(f"  Temperature: {weather_data['temperature']}°C")
        print(f"  Humidity: {weather_data['humidity']}%")
        print(f"  Monthly Rainfall: {weather_data['rainfall']}mm")
        
        return weather_data

    except Exception as e:
        print("✗ Weather API error:")
        print(f"  Error details: {str(e)}")
        # Return default values if API fails
        return {
            'temperature': 25,
            'humidity': 75,
            'rainfall': 200.0
        }
# Soil ranges dictionary remains the same
SOIL_RANGES = {
    'clay': {
        'temperature': (20, 25),
        'humidity': (80, 85),
        'rainfall': (200, 250)
    },
    'loam': {
        'temperature': (21, 26),
        'humidity': (80, 84),
        'rainfall': (220, 270)
    },
    'sandy': {
        'temperature': (22, 27),
        'humidity': (80, 83),
        'rainfall': (180, 240)
    },
    'gravel': {
        'temperature': (20, 25),
        'humidity': (81, 84),
        'rainfall': (230, 280)
    },
    'slit': {
        'temperature': (21, 26),
        'humidity': (80, 83),
        'rainfall': (200, 260)
    }
}