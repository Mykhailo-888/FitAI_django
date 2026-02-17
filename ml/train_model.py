import pandas as pd
import numpy as np
from fit_model_core import FitnessNeuralNet

print("Запуск навчання...")

# ===== LOAD DATA =====
df = pd.read_csv("data/gym_members_exercise_tracking.csv")
df = df.loc[:, ~df.columns.duplicated()]

# ===== REAL FEATURES ONLY =====
features = [
    "Age",
    "Height (m)",
    "Weight (kg)",
    "Avg_BPM",
    "Session_Duration (hours)",
    "Workout_Frequency (days/week)"
]

target = "Calories_Burned"

df = df[features + [target]].dropna()

X = df[features].values
y = df[target].values.reshape(-1, 1)

# ===== NORMALIZATION =====
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
y = (y - y.mean()) / (y.std() + 1e-8)

print("Розмір даних:", X.shape, y.shape)

# ===== MODEL =====
model = FitnessNeuralNet(
    lr=0.01,
    n_iters=5000,
    hidden_sizes=[16, 8]
)

print("Викликаємо fit...")
model.fit(X, y)

print("Після fit:")
print("Кількість вагових матриць:", len(model.weights))
print("Форма першої ваги:", model.weights[0].shape)
print("Форма останньої ваги:", model.weights[-1].shape)

# ===== SAVE =====
model.save_model("models/trained_fitness_model.pkl")
print("Модель збережена!")
