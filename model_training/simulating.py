import pandas as pd
import numpy as np

def generate_land_size(crop_type):
    """Generate realistic land size in hectares based on crop type"""
    if crop_type in ['Maize', 'Tea', 'Coconut']:
        # Larger land sizes
        return np.random.uniform(1.0, 10.0)
    elif crop_type in ['Green Gram', 'Green Beans', 'Cowpea', 'Chili', 'Mungbean', 'Blackgram']:
        # Medium land sizes for vegetables
        return np.random.uniform(0.2, 2.0)
    else:
        # Smaller land sizes for fruits and others
        return np.random.uniform(0.1, 1.0)

def generate_skewed_rainfall():
    # Generate a random number between 0 and 1
    r = np.random.random()
    
    # Apply different ranges with different probabilities
    if r < 0.4:  # 40% chance of very low rainfall
        return np.random.uniform(1, 10)
    elif r < 0.7:  # 30% chance of low rainfall
        return np.random.uniform(10, 50)
    elif r < 0.9:  # 20% chance of moderate rainfall
        return np.random.uniform(50, 100)
    else:  # 10% chance of high rainfall
        return np.random.uniform(100, 200)

# Read original dataset
df = pd.read_csv('model_training/crop data.csv')

# Define the crop mapping
crop_mapping = {
    'maize': 'Maize',
    'chickpea': 'chickpea',
    'kidneybeans': 'Cowpea',
    'pigeonpeas': 'Green Gram',
    'mothbeans': 'Cowpea',
    'blackgram': 'Blackgram',
    'lentil': 'Cowpea',
    'pomegranate': 'Pomegranate',
    'banana': 'Banana',
    'mango': 'Mango',
    'grapes': 'Papaya',
    'watermelon': 'Watermelon',
    'muskmelon': 'Pumpkin',
    'apple': 'Pineapple',
    'orange': 'Orange',
    'papaya': 'Papaya',
    'coconut': 'Coconut',
    'cotton': 'Chili',
    'jute': 'Coconut',
    'coffee': 'Tea'
}

# Convert labels to lowercase and apply mapping
df['label'] = df['label'].str.lower()
df['label'] = df['label'].map(crop_mapping)

# Add new columns
# Land size based on crop type
df['land_size'] = df['label'].apply(generate_land_size)

# Generate skewed rainfall data
df['rainfall'] = [generate_skewed_rainfall() for _ in range(len(df))]

# Water supply (more likely to be Yes for certain crops)
def generate_water_supply(row):
    crop = row['label']
    # Crops that typically need more water
    high_water_crops = ['Green Beans', 'Watermelon', 'Papaya', 'Banana', 'Tea']
    if crop in high_water_crops:
        return np.random.choice([0, 1], p=[0.2, 0.8])  # 80% chance of water supply
    else:
        return np.random.choice([0, 1], p=[0.4, 0.6])  # 60% chance of water supply

df['water_supply'] = df.apply(generate_water_supply, axis=1)

# Define crop categories
cereals = ['Maize']
vegetables = ['Green Gram', 'Green Beans', 'Cowpea', 'Chili', 'Mungbean', 'Blackgram']
fruits = ['Pomegranate', 'Banana', 'Mango', 'Papaya', 'Watermelon', 'Pumpkin', 'Pineapple', 'Orange', 'Coconut']
other = ['Tea']

# Generate preferences based on crop type
df['preference_cereals'] = df['label'].isin(cereals)
df['preference_vegetables'] = df['label'].isin(vegetables)
df['preference_fruits'] = df['label'].isin(fruits)

# Convert boolean preferences to int
df['preference_cereals'] = df['preference_cereals'].astype(int)
df['preference_vegetables'] = df['preference_vegetables'].astype(int)
df['preference_fruits'] = df['preference_fruits'].astype(int)

# Adjust N, P, K values based on crop type (if needed)
def adjust_npk(row):
    crop = row['label']
    # Example adjustments (you might want to fine-tune these)
    if crop in vegetables:
        row['N'] *= np.random.uniform(0.9, 1.1)
        row['P'] *= np.random.uniform(0.9, 1.1)
        row['K'] *= np.random.uniform(0.9, 1.1)
    elif crop in fruits:
        row['N'] *= np.random.uniform(0.8, 1.2)
        row['P'] *= np.random.uniform(0.8, 1.2)
        row['K'] *= np.random.uniform(0.8, 1.2)
    return row

df = df.apply(adjust_npk, axis=1)

# Ensure all values are within reasonable ranges
df['N'] = df['N'].clip(0, 140)
df['P'] = df['P'].clip(0, 145)
df['K'] = df['K'].clip(0, 205)
df['temperature'] = df['temperature'].clip(8, 45)
df['humidity'] = df['humidity'].clip(14, 100)
df['ph'] = df['ph'].clip(3.5, 10)

# Round numerical values to reasonable decimals
df['N'] = df['N'].round(2)
df['P'] = df['P'].round(2)
df['K'] = df['K'].round(2)
df['temperature'] = df['temperature'].round(2)
df['humidity'] = df['humidity'].round(2)
df['ph'] = df['ph'].round(2)
df['rainfall'] = df['rainfall'].round(2)
df['land_size'] = df['land_size'].round(3)

df.dropna(inplace=True)

# Save the updated dataset
df.to_csv('model_training/updated_crop_data.csv', index=False)

# Print statistics to verify the data
print("\nDataset Statistics:")
print(f"Total number of records: {len(df)}")
print("\nCrop distribution:")
print(df['label'].value_counts())
print("\nAverage land size by crop type:")
print(df.groupby('label')['land_size'].mean().sort_values(ascending=False))
print("\nWater supply distribution:")
print(df['water_supply'].value_counts(normalize=True))
print("\nRainfall distribution statistics:")
print(f"Mean rainfall: {df['rainfall'].mean():.2f}")
print(f"Median rainfall: {df['rainfall'].median():.2f}")
print("\nRainfall percentiles:")
print(df['rainfall'].describe([0.1, 0.25, 0.5, 0.75, 0.9]))