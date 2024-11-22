# In soil_ranges.py
SOIL_RANGES = {
    'clay': {
        'N': (60, 95),
        'P': (35, 55),
        'K': (35, 45),
        'ph': (6.0, 7.5)
    },
    'loam': {
        'N': (70, 90),
        'P': (40, 60),
        'K': (35, 45),
        'ph': (6.5, 7.8)
    },
    'sandy': {
        'N': (60, 85),
        'P': (35, 50),
        'K': (35, 42),
        'ph': (5.7, 6.8)
    },
    'gravel': {
        'N': (75, 95),
        'P': (45, 58),
        'K': (38, 44),
        'ph': (6.8, 7.8)
    },
    'slit': {
        'N': (65, 85),
        'P': (35, 55),
        'K': (36, 42),
        'ph': (6.0, 7.2)
    }
}

def get_soil_values(soil_type):
    """
    Get consistent values for soil parameters (N, P, K, pH) based on soil type.
    Uses the middle point of ranges for consistency.
    """
    if soil_type not in SOIL_RANGES:
        raise ValueError(f"Invalid soil type: {soil_type}. Valid types are: {', '.join(SOIL_RANGES.keys())}")
    
    ranges = SOIL_RANGES[soil_type]
    return {
        'N': round((ranges['N'][0] + ranges['N'][1]) / 2, 2),  # Use middle point
        'P': round((ranges['P'][0] + ranges['P'][1]) / 2, 2),
        'K': round((ranges['K'][0] + ranges['K'][1]) / 2, 2),
        'ph': round((ranges['ph'][0] + ranges['ph'][1]) / 2, 3)
    }