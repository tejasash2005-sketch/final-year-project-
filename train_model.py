import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

os.makedirs("models", exist_ok=True)

# Dummy training data
X = np.random.rand(500, 29)
y = np.random.randint(0,2,500)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier()
model.fit(X_scaled, y)

pickle.dump(model, open("models/rf_model.pkl","wb"))
pickle.dump(scaler, open("models/scaler.pkl","wb"))

print("✅ Model Created")