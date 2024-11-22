import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("Loading and preparing dataset...")
df = pd.read_csv('model_training/crop data.csv')

# Remove rice samples and shuffle
df = df[df['label'] != 'rice'].sample(frac=1, random_state=42).reset_index(drop=True)

print("\nClass Distribution after removing rice:")
print(df['label'].value_counts())

# Separate features and target
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Create and fit the LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Convert to categorical (one-hot encoding)
y_categorical = tf.keras.utils.to_categorical(y_encoded)

print("\nUnique crops after encoding:")
for i, crop in enumerate(le.classes_):
    print(f"{i}: {crop}")

# Create and fit the StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_categorical,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Calculate class weights for balanced training
class_weights = {}
for i in range(len(le.classes_)):
    class_weights[i] = len(y_encoded) / (len(le.classes_) * np.sum(y_encoded == i))

# Build the model
def build_model(input_shape, num_classes):
    model = Sequential([
        # Input layer
        Dense(128, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        
        # Hidden layers
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    return model

# Create and compile the model
model = build_model(input_shape=(X_train.shape[1],), num_classes=len(le.classes_))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks for better training
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    )
]

# Train the model
print("\nTraining deep learning model...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# Evaluate the model
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

print("\nModel Performance:")
print("Accuracy:", accuracy_score(y_test_labels, y_pred))
print("\nClassification Report:")
print(classification_report(y_test_labels, y_pred, target_names=le.classes_))

# Save the model components
print("\nSaving model components...")
os.makedirs('backend/models', exist_ok=True)

# Save Keras model with .keras extension
model.save('backend/models/crop_prediction_model.keras')

# Save scaler and label encoder
with open('backend/models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('backend/models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("Model components saved successfully.")

# Test predictions with different conditions
print("\nTesting predictions with different conditions...")
test_conditions = [
    {
        'N': 90, 'P': 40, 'K': 40,
        'temperature': 20, 'humidity': 82,
        'ph': 6.5, 'rainfall': 200
    },
    {
        'N': 70, 'P': 35, 'K': 45,
        'temperature': 25, 'humidity': 70,
        'ph': 7.0, 'rainfall': 150
    },
    {
        'N': 120, 'P': 45, 'K': 50,
        'temperature': 30, 'humidity': 85,
        'ph': 6.8, 'rainfall': 250
    }
]

print("\nPredictions for test conditions:")
for i, conditions in enumerate(test_conditions, 1):
    # Create a DataFrame with the test conditions
    test_df = pd.DataFrame([conditions])
    
    # Scale the features
    test_scaled = scaler.transform(test_df)
    
    # Get prediction probabilities
    proba = model.predict(test_scaled)[0]
    
    # Get top 3 predictions
    top_3_idx = np.argsort(proba)[-3:][::-1]
    
    print(f"\nTest Case {i}:")
    print("Conditions:", conditions)
    print("Top 3 predictions:")
    for idx in top_3_idx:
        print(f"{le.classes_[idx]}: {proba[idx]:.2%}")

# Save model summary
with open('backend/models/model_summary.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

# Save training history
history_dict = history.history
with open('backend/models/training_history.pkl', 'wb') as f:
    pickle.dump(history_dict, f)