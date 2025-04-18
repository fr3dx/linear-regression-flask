from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('regression_model.pkl')

@app.route('/')
def home():
    # Render the index.html file when the browser requests the root URL
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read JSON data from the request
        data = request.get_json()

        # Extract input features
        x1 = float(data['features'][0])  # First feature
        x2 = float(data['features'][1])  # Second feature
        
        input_data = np.array([[x1, x2]])
        prediction = model.predict(input_data)
        prediction = np.round(prediction, 6)

        # Return the prediction in JSON format
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



