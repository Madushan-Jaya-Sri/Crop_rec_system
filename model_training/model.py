import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data():
    print("Loading and preparing data...")
    df = pd.read_csv('model_training/updated_crop_data.csv')
    
    # Create synthetic samples focusing on weather variations
    weather_samples = []
    for _, row in df.iterrows():
        # Create multiple variations for each row with different weather conditions
        for _ in range(5):
            new_row = row.copy()
            # Vary weather conditions
            new_row['temperature'] = row['temperature'] * np.random.uniform(0.6, 1.4)
            new_row['humidity'] = row['humidity'] * np.random.uniform(0.6, 1.4)
            new_row['rainfall'] = row['rainfall'] * np.random.uniform(0.6, 1.4)
            
            # Set random preference values (making them irrelevant)
            new_row['preference_cereals'] = np.random.choice([0, 1])
            new_row['preference_vegetables'] = np.random.choice([0, 1])
            new_row['preference_fruits'] = np.random.choice([0, 1])
            
            weather_samples.append(new_row)
    
    weather_df = pd.DataFrame(weather_samples)
    df = pd.concat([df, weather_df], ignore_index=True)
    
    # Create weather interaction features
    df['weather_score'] = (df['temperature'] * df['humidity'] * df['rainfall']) / 1000
    df['water_availability'] = df['rainfall'] * df['water_supply']
    
    # Separate features and target
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 
            'land_size', 'water_supply', 'preference_cereals', 
            'preference_vegetables', 'preference_fruits',
            'weather_score', 'water_availability']]
    y = df['label']
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return X, y_encoded, le

def preprocess_data(X, y):
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train, y_train):
    print("Training model...")
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=3,
        subsample=0.7,
        colsample_bytree=0.7,
        gamma=2,  # Higher regularization
        random_state=42,
        use_label_encoder=False
    )
    
    # Create sample weights
    sample_weights = np.ones(len(y_train))
    
    # Higher weights for samples with significant weather variations
    for i in range(len(X_train)):
        weather_weight = 1.0
        
        # Check if weather features are significantly different from mean
        if abs(X_train.iloc[i]['temperature']) > 1.0:
            weather_weight *= 2.0
        if abs(X_train.iloc[i]['humidity']) > 1.0:
            weather_weight *= 2.0
        if abs(X_train.iloc[i]['rainfall']) > 1.0:
            weather_weight *= 2.0
            
        # Reduce weight if preferences are involved
        if (X_train.iloc[i]['preference_cereals'] == 1 or 
            X_train.iloc[i]['preference_vegetables'] == 1 or 
            X_train.iloc[i]['preference_fruits'] == 1):
            weather_weight *= 0.01
            
        sample_weights[i] = weather_weight
    
    # Normalize sample weights
    sample_weights = sample_weights / sample_weights.mean()
    
    # Train with sample weights
    model.fit(
        X_train, 
        y_train,
        sample_weight=sample_weights
    )
    
    return model

def evaluate_model(model, X_test, y_test, le):
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def analyze_feature_importance(model, X):
    print("\nAnalyzing feature importance...")
    importance_scores = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    print("\nFeature Importance:")
    print(feature_importance)
    return feature_importance

def save_model(model, scaler, le):
    print("\nSaving model and related objects...")
    
    # Create models directory if it doesn't exist
    import os
    os.makedirs('./backend/models', exist_ok=True)
    
    # Save model and related objects
    with open('./backend/models/xgboost_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('./backend/models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('./backend/models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

def test_preference_invariance(model, scaler, le, X_test):
    print("\nTesting prediction invariance to preferences...")
    
    # Take a sample row
    sample = X_test.iloc[0:1].copy()
    original_prediction = model.predict(sample)[0]
    
    # Test with different preference combinations
    preference_combinations = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1]
    ]
    
    predictions = []
    for prefs in preference_combinations:
        test_sample = sample.copy()
        test_sample['preference_cereals'] = prefs[0]
        test_sample['preference_vegetables'] = prefs[1]
        test_sample['preference_fruits'] = prefs[2]
        pred = model.predict(test_sample)[0]
        predictions.append(pred)
        
    unique_predictions = len(set(predictions))
    print(f"Number of unique predictions across preference combinations: {unique_predictions}")
    if unique_predictions == 1:
        print("✓ Model is invariant to preference changes!")
    else:
        print("! Model shows some dependence on preferences")

def main():
    # Load and prepare data
    X, y, le = load_and_prepare_data()
    
    # Preprocess data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(X, y)
    
    # Train model
    model = train_model(X_train_scaled, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test_scaled, y_test, le)
    
    # Analyze feature importance
    analyze_feature_importance(model, X)
    
    # Test preference invariance
    test_preference_invariance(model, scaler, le, X_test_scaled)
    
    # Save model and related objects
    save_model(model, scaler, le)

if __name__ == "__main__":
    main()