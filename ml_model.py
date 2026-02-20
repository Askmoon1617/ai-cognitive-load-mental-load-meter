import numpy as np
from sklearn.linear_model import LogisticRegression

# Dummy training data (simulated cognitive states)
# [blink_rate, ear_mean, ear_std, duration]
X = np.array([
    [0.1, 0.30, 0.01, 10],  # LOW
    [0.15, 0.28, 0.02, 15],
    [0.25, 0.25, 0.04, 20],  # MODERATE
    [0.3, 0.23, 0.05, 25],
    [0.45, 0.20, 0.08, 30],  # HIGH
    [0.5, 0.18, 0.10, 35],
])

# Labels: 0 = Low, 1 = Moderate, 2 = High
y = np.array([0, 0, 1, 1, 2, 2])

model = LogisticRegression()
model.fit(X, y)

def predict_load(blink_rate, ear_mean, ear_std, duration):
    features = np.array([[blink_rate, ear_mean, ear_std, duration]])
    return model.predict(features)[0]