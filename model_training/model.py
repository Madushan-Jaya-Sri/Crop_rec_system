import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
def load_and_prepare_data():
    print("Loading and preparing data...")
    df = pd.read_csv('model_training/updated_crop_data.csv')
    
    # Separate features and target
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 
            'land_size', 'water_supply', 'preference_cereals', 
            'preference_vegetables', 'preference_fruits']]
    y = df['label']
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return X, y_encoded, le

# In your model training script
def preprocess_data(X, y):
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create scaler with feature names
    scaler = StandardScaler()
    # Fit and transform while preserving feature names
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Train model
def train_model(X_train, y_train):
    print("Training model...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        use_label_encoder=False
    )
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train the final model
    model.fit(X_train, y_train)
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test, le):
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
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

# Analyze feature importance
def analyze_feature_importance(model, X):
    print("\nAnalyzing feature importance...")
    importance_scores = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    print("\nFeature Importance:")
    print(feature_importance)
    return feature_importance

# Save model and related objects
def save_model(model, scaler, le):
    print("\nSaving model and related objects...")
    with open('./backend/models/xgboost_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('./backend/models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('./backend/models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

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
    
    # Save model and related objects
    save_model(model, scaler, le)
    
    # Test model prediction
    print("\nTesting model prediction...")
    # Create a sample input
    sample_input = np.array([[60, 55, 44, 23.004459, 82.320763, 7.840207, 263.964248, 
                             2.5, 1, 0, 1, 0]])  # Example values
    sample_scaled = scaler.transform(sample_input)
    prediction = model.predict(sample_scaled)
    predicted_crop = le.inverse_transform(prediction)[0]
    print(f"Sample prediction: {predicted_crop}")

if __name__ == "__main__":
    main()