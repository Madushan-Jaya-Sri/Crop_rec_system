from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the saved model and label encoder
loaded_model = joblib.load('xgboost_model.pkl')
loaded_label_encoder = joblib.load('label_encoder.pkl')

# Load dataset
df = pd.read_csv('/Users/jayasri/Desktop/crop data.csv')
df = df[~(df['label']=='rice')]

@app.route('/')
def home():
    # Show sample data on home page
    return render_template('index.html', tables=[df.head().to_html(classes='data', header="true")])

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from form
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        
        # Create DataFrame for prediction
        new_data = pd.DataFrame({
            'N': [N],
            'P': [P],
            'K': [K],
            'temperature': [temperature],
            'humidity': [humidity],
            'ph': [ph],
            'rainfall': [rainfall]
        })
        
        # Predict using the loaded model
        new_pred = loaded_model.predict(new_data)

        # Decode the prediction from numerical labels to original labels
        decoded_pred = loaded_label_encoder.inverse_transform(new_pred)

        # Display result
        return render_template('result.html', prediction=decoded_pred[0])

if __name__ == '__main__':
    app.run(debug=True)
