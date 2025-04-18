import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Input data (features) and corresponding outputs (targets)
X = np.array([
    [1.0, 2.0],
    [2.0, 1.0],
    [0.0, 3.0],
    [3.0, 0.0],
    [1.0, 1.0],
    [4.0, 1.0]
])

Y = np.array([
    [7, -1, 7],
    [8, 1, 2],
    [6, -3, 12],
    [9, 3, -3],
    [5, 0, 3],
    [14, 3, 0]
])

# Create and train the linear regression model
model = LinearRegression()
model.fit(X, Y)

# Save the trained model to a file
joblib.dump(model, 'regression_model.pkl')

