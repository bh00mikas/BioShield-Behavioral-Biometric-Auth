
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def simulate_mouse_data(user_id, n_points=300):
    x = np.cumsum(np.random.randn(n_points))
    y = np.cumsum(np.random.randn(n_points))
    timestamps = np.cumsum(np.abs(np.random.randn(n_points)))
    df = pd.DataFrame({
        'user': user_id,
        'x': x,
        'y': y,
        'timestamp': timestamps
    })
    return df

user1 = simulate_mouse_data('user1')
user2 = simulate_mouse_data('user2')
data = pd.concat([user1, user2], ignore_index=True)

data['dx'] = data.groupby('user')['x'].diff().fillna(0)
data['dy'] = data.groupby('user')['y'].diff().fillna(0)
data['dt'] = data.groupby('user')['timestamp'].diff().fillna(1)
data['velocity'] = np.sqrt(data['dx']**2 + data['dy']**2) / data['dt']
data['acceleration'] = data.groupby('user')['velocity'].diff().fillna(0) / data['dt']

features = data.groupby('user').agg({
    'velocity': ['mean', 'std'],
    'acceleration': ['mean', 'std']
})
features.columns = ['_'.join(col) for col in features.columns]
features = features.reset_index()

X = features.drop('user', axis=1)
y = features['user']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/mouse_auth.pkl")

print("✅ Model trained and saved.")
