from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Betöltjük a már betanított modellt
model = joblib.load('regression_model.pkl')

@app.route('/')
def home():
    # Az index.html fájl renderelése, amit a böngésző kér
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # JSON adat beolvasása
        data = request.get_json()

        # Kivesszük a bemeneti adatokat
        x1 = float(data['features'][0])  # Az első feature
        x2 = float(data['features'][1])  # A második feature
        
        input_data = np.array([[x1, x2]])
        prediction = model.predict(input_data)
        prediction = np.round(prediction, 6)

        # Visszaadjuk a predikciót JSON formátumban
        return jsonify({
            'y1': prediction[0][0],
            'y2': prediction[0][1],
            'y3': prediction[0][2],
            'x1': x1,
            'x2': x2
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)



# Invoke-RestMethod -Method POST -Uri http://127.0.0.1:5000/predict -Body (@{features = @(4.0, 1.0)} | ConvertTo-Json -Depth 2) -ContentType "application/json"
