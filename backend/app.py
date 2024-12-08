from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import os
import traceback
import requests
from dotenv import load_dotenv
from location_weather import get_current_location, get_coordinates_from_city, get_weather
from utils.soil_ranges import get_soil_values

load_dotenv(override=True)
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')
OPENCAGE_API_KEY = os.getenv('OPENCAGE_API_KEY')
app = Flask(__name__)
CORS(app)

# Updated feature names to match the new model
FEATURE_NAMES = [
    'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall',
    'land_size', 'water_supply', 'preference_cereals',
    'preference_vegetables', 'preference_fruits',
    'weather_score', 'water_availability'
]

# Get the absolute path to the models directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def load_models():
    try:
        model_path = os.path.join(MODELS_DIR, 'xgboost_model.pkl')
        scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
        le_path = os.path.join(MODELS_DIR, 'label_encoder.pkl')

        print("\n=== Debug Point 1: Loading Models ===")
        print(f"Looking for model at: {model_path}")
        print(f"Looking for scaler at: {scaler_path}")
        print(f"Looking for label encoder at: {le_path}")
        
        if not all(os.path.exists(path) for path in [model_path, scaler_path, le_path]):
            raise FileNotFoundError("One or more model files are missing")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(le_path, 'rb') as f:
            le = pickle.load(f)
            
        print("Successfully loaded all model components")
        return model, scaler, le
    except Exception as e:
        print("\n=== Error Loading Models ===")
        print(f"Error details: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        raise

os.makedirs(MODELS_DIR, exist_ok=True)

try:
    model, scaler, le = load_models()
except Exception as e:
    print(f"Failed to load models: {str(e)}")
    raise

@app.route('/api/predict', methods=['GET', 'POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204
    elif request.method == 'GET':
        return jsonify({"message": "API is running. Please use POST method for predictions."})
        
    try:
        data = request.json
        print("\n=== Debug Point 1: Incoming Request Data ===")
        print("Request Data:", data)

        # Get location and weather data
        print("\n=== Debug Point 2: Processing Location ===")
        if data['locationType'] == 'automatic':
            print("Getting current location automatically...")
            city, lat, lon = get_current_location()
            if city:
                print(f"✓ Current location detected: {city} ({lat}, {lon})")
            else:
                print("✗ Failed to get current location, using defaults")
                lat, lon = None, None
        else:
            print(f"Getting coordinates for city: {data['location']}")
            lat, lon = get_coordinates_from_city(data['location'], OPENCAGE_API_KEY)
            city = data['location']
            if lat and lon:
                print(f"✓ Coordinates found: {lat}, {lon}")
            else:
                print("✗ Failed to get coordinates, using defaults")

        # Get weather data
        weather_data = get_weather(lat, lon, WEATHER_API_KEY)
        if not weather_data:
            weather_data = {
                'temperature': 25,
                'humidity': 75,
                'rainfall': 200
            }

        # Get soil values based on soil type
        try:
            soil_values = get_soil_values(data['soilType'])
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Error with soil type: {str(e)}'
            }), 400
        
        # Calculate additional features
        weather_score = (weather_data['temperature'] * weather_data['humidity'] * weather_data['rainfall']) / 1000
        water_availability = weather_data['rainfall'] * data['water_supply']
        
        input_data = pd.DataFrame([[
            soil_values['N'],
            soil_values['P'],
            soil_values['K'],
            weather_data['temperature'],
            weather_data['humidity'],
            soil_values['ph'],
            weather_data['rainfall'],
            data['land_size'],
            data['water_supply'],
            data['preference_cereals'],
            data['preference_vegetables'],
            data['preference_fruits'],
            weather_score,
            water_availability
        ]], columns=FEATURE_NAMES)
        
        # Scale features
        features_scaled = pd.DataFrame(
            scaler.transform(input_data),
            columns=FEATURE_NAMES
        )
        
        # Get prediction probabilities
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Get top 3 predictions with their probabilities
        top_3_idx = np.argsort(probabilities)[-3:][::-1]
        top_3_crops = le.inverse_transform(top_3_idx)
        top_3_probas = probabilities[top_3_idx]
        
        # Create predictions list
        predictions = []
        for crop, prob in zip(top_3_crops, top_3_probas):
            predictions.append({
                'crop': crop,
                'probability': float(prob * 100)
            })
        
        response_data = {
            'status': 'success',
            'predictions': predictions,
            'weather': {
                'temperature': weather_data['temperature'],
                'humidity': weather_data['humidity'],
                'rainfall': weather_data['rainfall'],
                'location': city if city else "Unknown"
            },
            'soil_values': soil_values,
            'model_inputs': input_data.to_dict('records')[0]
        }
        
        print("\n=== Debug Point 8: Final Response ===")
        print("Sending response:", response_data)
        return jsonify(response_data)
        
    except Exception as e:
        print("\n=== Error Occurred ===")
        print("Error details:", str(e))
        print("Full traceback:")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e),
            'details': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)