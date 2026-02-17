import pandas as pd
import numpy as np

print("=== Створюємо новий датасет з твоїми 23 параметрами ===")

# Завантажуємо оригінальний файл
df = pd.read_csv(r"C:\FitAI_django\data\gym_members_exercise_tracking.csv")

# Твої 23 параметри (без фото, бо фото не число)
params_23 = [
    "Вік", "Зріст", "Вага", "Обхват талії", "Емоційний стрес",
    "Алкоголь (одиниць/тиждень)", "Калораж (ккал/день)",
    "Віджимання max", "Підтягування max", "Біг 1 км", "Біг 100 м",
    "Тест Купера", "Бурпі за 3 хв", "Віджимання за 1 хв", "Сон",
    "Пульс спокою", "Тиск верхній",
    "Мітохондрії (заглушка)", "Тестостерон", "Кортизол", "Гемоглобін", "СРБ"
]

# Створюємо нову порожню таблицю з твоїми колонками
df_new = pd.DataFrame(index=df.index, columns=params_23)

# Копіюємо реальні дані з оригіналу
df_new["Вік"] = df["Age"]
df_new["Зріст"] = df["Height (m)"] * 100           # метри → сантиметри
df_new["Вага"] = df["Weight (kg)"]
df_new["Обхват талії"] = df["Fat_Percentage"] * 2  # умовне наближення
df_new["Пульс спокою"] = df["Resting_BPM"]
df_new["Сон"] = df["Session_Duration (hours)"]
df_new["Калораж (ккал/день)"] = df["Calories_Burned"] / df["Workout_Frequency (days/week)"]

# Заповнюємо решту колонок випадковими реалістичними значеннями
np.random.seed(42)  # щоб було відтворювано

df_new["Емоційний стрес"] = np.random.randint(1, 11, len(df))           # 1–10
df_new["Алкоголь (одиниць/тиждень)"] = np.random.randint(0, 21, len(df))  # 0–20
df_new["Віджимання max"] = np.random.randint(5, 51, len(df))           # 5–50
df_new["Підтягування max"] = np.random.randint(0, 31, len(df))         # 0–30
df_new["Біг 1 км"] = np.random.uniform(4.5, 10.0, len(df)).round(1)    # 4.5–10.0 хв
df_new["Біг 100 м"] = np.random.uniform(12.0, 18.0, len(df)).round(1)  # 12–18 сек
df_new["Тест Купера"] = np.random.uniform(2.0, 4.0, len(df)).round(1)  # 2–4 км
df_new["Бурпі за 3 хв"] = np.random.randint(20, 81, len(df))           # 20–80
df_new["Віджимання за 1 хв"] = np.random.randint(15, 61, len(df))      # 15–60
df_new["Тиск верхній"] = np.random.randint(100, 151, len(df))          # 100–150 мм рт.ст.
df_new["Мітохондрії (заглушка)"] = np.random.randint(0, 101, len(df))  # 0–100
df_new["Тестостерон"] = np.random.randint(300, 901, len(df))           # 300–900 нг/дл
df_new["Кортизол"] = np.random.uniform(8.0, 22.0, len(df)).round(1)    # 8–22 мкг/дл
df_new["Гемоглобін"] = np.random.uniform(12.0, 17.0, len(df)).round(1) # 12–17 г/дл
df_new["СРБ"] = np.random.uniform(0.5, 3.0, len(df)).round(1)          # 0.5–3 мг/л

# Заповнюємо пропуски середніми (якщо щось пропустили)
df_new = df_new.fillna(df_new.mean(numeric_only=True))

# Зберігаємо новий файл
df_new.to_csv(r"C:\FitAI_django\data\edited_23_params.csv", index=False)

print("Успішно!")
print("Новий файл збережено:", "data/edited_23_params.csv")
print("Розмір нового датасету:", df_new.shape)
print("\nПерші 5 рядків (щоб побачив різноманітність):\n")
print(df_new.head())