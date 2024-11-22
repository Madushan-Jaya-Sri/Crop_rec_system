import os
import requests
from geopy.geocoders import Nominatim
from dotenv import load_dotenv

load_dotenv()

def get_current_location():
    """
    Get the current location using multiple IP geolocation services.
    Returns city name, latitude and longitude.
    """
    try:
        # First attempt: using ip-api.com (free, no API key required)
        response = requests.get('http://ip-api.com/json')
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                return data['city'], data['lat'], data['lon']

        # Second attempt: using ipapi.co with a different endpoint
        response = requests.get('https://ipapi.co/json/', timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get('city'), data.get('latitude'), data.get('longitude')

        # Third attempt: using ipinfo.io (works without API key but has rate limits)
        response = requests.get('https://ipinfo.io/json')
        if response.status_code == 200:
            data = response.json()
            if 'loc' in data:
                lat, lon = data['loc'].split(',')
                return data.get('city'), float(lat), float(lon)

        # Fallback: using extreme-ip-lookup.com
        response = requests.get('https://extreme-ip-lookup.com/json/')
        if response.status_code == 200:
            data = response.json()
            return data.get('city'), float(data.get('lat')), float(data.get('lon'))

        raise Exception("Could not determine location using any available service")

    except Exception as e:
        print(f"Error getting location: {e}")
        return None, None, None

def get_coordinates_from_city(city_name):
    """
    Get coordinates from city name using Nominatim geocoder.
    Returns latitude and longitude.
    """
    try:
        geolocator = Nominatim(user_agent="my_weather_app")
        location = geolocator.geocode(city_name)
        if location:
            return location.latitude, location.longitude
        return None, None
    except Exception as e:
        print(f"Error getting coordinates: {e}")
        return None, None

def get_weather(query, api_key):
    """
    Get weather data using Weatherstack API.
    Accepts either city name or coordinates as query.
    """
    try:
        url = "http://api.weatherstack.com/current"
        params = {
            "access_key": api_key,
            "query": query
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        # Check for API errors
        if "error" in data:
            print(f"API Error: {data['error']['info']}")
            return None
            
        return {
            'location': {
                'name': data['location']['name'],
                'country': data['location']['country'],
                'lat': data['location']['lat'],
                'lon': data['location']['lon']
            },
            'current': {
                'temp_c': data['current']['temperature'],
                'humidity': data['current']['humidity'],
                'rainfall': data['current']['precip']  # Changed key name to rainfall but still using precip data
            }
        }
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return None

def main():
    """
    Main function to run the weather application.
    Allows user to choose between current location or manual city entry.
    """
    api_key = os.getenv("WEATHERSTACK_API_KEY")
    if not api_key:
        print("Please set your WEATHERSTACK_API_KEY environment variable")
        return

    print("\nWeather Information Service")
    print("1. Use current location")
    print("2. Enter city manually")
    
    try:
        choice = input("\nEnter your choice (1 or 2): ")
        
        if choice == "1":
            print("\nGetting current location...")
            city, lat, lon = get_current_location()
            if city and lat and lon:
                query = f"{lat},{lon}"
                print(f"Located in: {city}")
            else:
                print("Could not determine current location")
                return
                
        elif choice == "2":
            city = input("\nEnter city name: ")
            query = city
        else:
            print("Invalid choice")
            return
            
        print("\nFetching weather data...")
        weather_data = get_weather(query, api_key)
        
        if weather_data:
            location = weather_data['location']
            current = weather_data['current']
            
            print(f"\nWeather in {location['name']}, {location['country']}:")
            print(f"Temperature: {current['temp_c']}°C")
            print(f"Humidity: {current['humidity']}%")
            print(f"Rainfall: {current['rainfall']} mm")  # Changed label from Precipitation to Rainfall
        else:
            print("Could not fetch weather data")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()