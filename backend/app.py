from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import os
import traceback
import requests
import random  # For soil ranges
from dotenv import load_dotenv
from location_weather import get_current_location, get_coordinates_from_city, get_weather

load_dotenv()
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')
OPENCAGE_API_KEY = os.getenv('OPENCAGE_API_KEY')
app = Flask(__name__)
CORS(app)

# Soil ranges dictionary
SOIL_RANGES = {
    'clay': {
        'N': (60, 95),
        'P': (35, 55),
        'K': (35, 45),
        'ph': (6.0, 7.5),
        'temperature': (20, 25),
        'humidity': (80, 85),
        'rainfall': (200, 250)
    },
    'loam': {
        'N': (70, 90),
        'P': (40, 60),
        'K': (35, 45),
        'ph': (6.5, 7.8),
        'temperature': (21, 26),
        'humidity': (80, 84),
        'rainfall': (220, 270)
    },
    'sandy': {
        'N': (60, 85),
        'P': (35, 50),
        'K': (35, 42),
        'ph': (5.7, 6.8),
        'temperature': (22, 27),
        'humidity': (80, 83),
        'rainfall': (180, 240)
    },
    'gravel': {
        'N': (75, 95),
        'P': (45, 58),
        'K': (38, 44),
        'ph': (6.8, 7.8),
        'temperature': (20, 25),
        'humidity': (81, 84),
        'rainfall': (230, 280)
    },
    'slit': {
        'N': (65, 85),
        'P': (35, 55),
        'K': (36, 42),
        'ph': (6.0, 7.2),
        'temperature': (21, 26),
        'humidity': (80, 83),
        'rainfall': (200, 260)
    }
}

# Define feature names
FEATURE_NAMES = [
    'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall',
    'land_size', 'water_supply', 'preference_cereals',
    'preference_vegetables', 'preference_fruits'
]

# Get the absolute path to the models directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def get_soil_values(soil_type):
    """Get soil parameter values based on soil type."""
    if soil_type not in SOIL_RANGES:
        raise ValueError(f"Invalid soil type: {soil_type}. Valid types are: {', '.join(SOIL_RANGES.keys())}")
    
    ranges = SOIL_RANGES[soil_type]
    return {
        'N': round(random.uniform(*ranges['N']), 2),
        'P': round(random.uniform(*ranges['P']), 2),
        'K': round(random.uniform(*ranges['K']), 2),
        'ph': round(random.uniform(*ranges['ph']), 3),
        'temperature': round(random.uniform(*ranges['temperature']), 2),
        'humidity': round(random.uniform(*ranges['humidity']), 2),
        'rainfall': round(random.uniform(*ranges['rainfall']), 2)
    }

def load_models():
    try:
        # Use os.path.join for platform-independent path handling
        model_path = os.path.join(MODELS_DIR, 'xgboost_model.pkl')
        scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
        le_path = os.path.join(MODELS_DIR, 'label_encoder.pkl')

        print("\n=== Debug Point 1: Loading Models ===")
        print(f"Looking for model at: {model_path}")
        print(f"Looking for scaler at: {scaler_path}")
        print(f"Looking for label encoder at: {le_path}")
        
        # Check if files exist
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

# Make sure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Load models
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

        # Get location and weather data based on user selection
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
        print("\n=== Debug Point 3: Fetching Weather Data ===")
        weather_data = get_weather(lat, lon, WEATHER_API_KEY)
        if weather_data:
            print("✓ Weather data received:")
            print(f"  Temperature: {weather_data['temperature']}°C")
            print(f"  Humidity: {weather_data['humidity']}%")
            print(f"  Rainfall: {weather_data['rainfall']}mm")
        else:
            print("✗ Using default weather values")
            weather_data = {
                'temperature': 25,
                'humidity': 75,
                'rainfall': 200
            }

        # Get soil values based on soil type
        try:
            soil_values = get_soil_values(data['soilType'])
            print("\n=== Debug Point 4: Soil Values ===")
            print("Soil Values:", soil_values)
        except Exception as e:
            print(f"Error getting soil values: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'Error with soil type: {str(e)}'
            }), 400
        
        input_data = pd.DataFrame([[
            soil_values['N'],           # From soil type
            soil_values['P'],           # From soil type
            soil_values['K'],           # From soil type
            weather_data['temperature'], # From weather API
            weather_data['humidity'],    # From weather API
            soil_values['ph'],          # From soil type
            weather_data['rainfall'],    # From weather API
            data['land_size'],
            data['water_supply'],
            data['preference_cereals'],
            data['preference_vegetables'],
            data['preference_fruits']
        ]], columns=FEATURE_NAMES)
        
        print("\n=== Debug Point 5: Input Data ===")
        print("Raw input data:")
        print(input_data)
        
        # Scale features
        features_scaled = pd.DataFrame(
            scaler.transform(input_data),
            columns=FEATURE_NAMES
        )
        
        print("\n=== Debug Point 6: Scaled Data ===")
        print("Scaled input data:")
        print(features_scaled)
        
        # Make prediction
        prediction = model.predict(features_scaled)
        predicted_crop = le.inverse_transform(prediction)[0]
        
        # Get prediction probabilities
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = float(max(probabilities) * 100)
        
        # Get top 3 predictions
        top_3_idx = np.argsort(probabilities)[-3:][::-1]
        top_3_crops = le.inverse_transform(top_3_idx)
        top_3_probas = probabilities[top_3_idx]
        
        print("\n=== Debug Point 7: Predictions ===")
        print(f"Predicted crop: {predicted_crop}")
        print(f"Confidence: {confidence}%")
        print("Top 3 predictions:", list(zip(top_3_crops, top_3_probas)))
        
        response_data = {
            'status': 'success',
            'prediction': predicted_crop,
            'confidence': round(confidence, 2),
            'top_3_predictions': [
                {'crop': crop, 'probability': float(prob)} 
                for crop, prob in zip(top_3_crops, top_3_probas)
            ],
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