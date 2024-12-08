# Soil values typical for Sri Lankan agricultural soils
SOIL_VALUES = {
    'clay': {
        'N': 85,  # High N content typical in Sri Lankan clay soils
        'P': 45,  # Moderate P level
        'K': 40,  # Moderate K level
        'ph': 6.8 # Slightly acidic to neutral, common in wet zone clay soils
    },
    'loam': {
        'N': 80,  # Good N content, typical for loamy soils in farming areas
        'P': 50,  # Moderate to high P content
        'K': 42,  # Good K level
        'ph': 7.0 # Neutral pH, ideal for most crops
    },
    'sandy': {
        'N': 65,  # Lower N content typical of sandy soils
        'P': 40,  # Moderate P level
        'K': 38,  # Moderate K level
        'ph': 6.2 # Slightly acidic, common in coastal sandy soils
    },
    'gravel': {
        'N': 78,  # Moderate to high N content
        'P': 48,  # Moderate P content
        'K': 40,  # Moderate K level
        'ph': 7.2 # Slightly alkaline
    },
    'slit': {
        'N': 70,  # Moderate N content
        'P': 45,  # Moderate P level
        'K': 39,  # Moderate K level
        'ph': 6.5 # Slightly acidic to neutral
    }
}

def get_soil_values(soil_type):
    """
    Get fixed values for soil parameters (N, P, K, pH) based on soil type.
    These values are typical for Sri Lankan agricultural soils.
    """
    if soil_type not in SOIL_VALUES:
        raise ValueError(f"Invalid soil type: {soil_type}. Valid types are: {', '.join(SOIL_VALUES.keys())}")
    
    return SOIL_VALUES[soil_type]