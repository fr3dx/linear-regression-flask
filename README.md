# Linear Regression Flask Application

This project is a simple web application built with Flask and Python. It demonstrates how to use a pre-trained linear regression model to make predictions based on user inputs. The model is trained to predict three variables, `y1`, `y2`, and `y3`, using two input features `x1` and `x2`.

## Project Description

The application is powered by Flask, a lightweight web framework in Python, and leverages a pre-trained linear regression model to predict values based on user input. Users can provide values for two features, `x1` and `x2`, and the app will return predicted values for `y1`, `y2`, and `y3`.

The model itself is based on the following three linear equations:

1. y1 = 3x₁ + 2x₂
2. y2 = x₁ - x₂
3. y3 = -x₁ + 4x₂

The project consists of two main components:
1. A Python model (`model.py`) that trains and saves the linear regression model.
2. A Flask web application (`app.py`) that loads the model and provides a web interface for users to make predictions.

## Project Structure

The project directory contains the following files:

- **`model.py`**: This script trains a linear regression model using sample data and saves the model to a file (`regression_model.pkl`).
- **`app.py`**: This is the main Flask application that serves the model to the frontend. It exposes a `/predict` endpoint to accept input and return predictions.
- **`static/style.css`**: A simple CSS file to style the frontend of the application.
- **`templates/index.html`**: The HTML frontend for the application, where users can input values for `x1` and `x2` to get predictions for `y1`, `y2`, and `y3`.
- **`regression_model.pkl`**: The pre-trained linear regression model saved in a pickle file.
- **`train_and_start_app.ps1`**: A PowerShell script to automate the process of training the model and starting the Flask app.

# Example POST request from PowerShell:
Invoke-RestMethod -Method POST -Uri http://127.0.0.1:5000/predict -Body (@{features = @(4.0, 1.0)} | ConvertTo-Json -Depth 2) -ContentType "application/json"

## How It Works

1. **Model Training** (`model.py`):
   The `model.py` file creates and trains a linear regression model using the following dataset:

   ```python
   X = [
       [1.0, 2.0],
       [2.0, 1.0],
       [0.0, 3.0],
       [3.0, 0.0],
       [1.0, 1.0],
       [4.0, 1.0]
   ]
   Y = [
       [7, -1, 7],
       [8, 1, 2],
       [6, -3, 12],
       [9, 3, -3],
       [5, 0, 3],
       [14, 3, 0]
   ]

