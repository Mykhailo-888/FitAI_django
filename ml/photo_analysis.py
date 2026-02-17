import cv2
import numpy as np
import os

def analyze_body_proportions(photo_path):
    if not os.path.exists(photo_path):
        return {"error": "Фото не знайдено"}

    img = cv2.imread(photo_path)
    if img is None:
        return {"error": "Не вдалося прочитати фото"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 30, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"error": "Не знайдено тіло"}

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Більш точне розділення
    shoulder_y_start = y + int(h * 0.1)
    shoulder_y_end = y + int(h * 0.25)
    shoulder_slice = edges[shoulder_y_start:shoulder_y_end, :]
    shoulder_contours, _ = cv2.findContours(shoulder_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shoulder_width = max(c.shape[1] for c in shoulder_contours) if shoulder_contours else w * 0.75

    waist_y_start = y + int(h * 0.35)
    waist_y_end = y + int(h * 0.5)
    waist_slice = edges[waist_y_start:waist_y_end, :]
    waist_contours, _ = cv2.findContours(waist_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    waist_width = max(c.shape[1] for c in waist_contours) if waist_contours else w * 0.55

    hip_y_start = y + int(h * 0.6)
    hip_y_end = y + int(h * 0.8)
    hip_slice = edges[hip_y_start:hip_y_end, :]
    hip_contours, _ = cv2.findContours(hip_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hip_width = max(c.shape[1] for c in hip_contours) if hip_contours else w * 0.85

    shoulder_waist_ratio = shoulder_width / waist_width if waist_width > 0 else 1.0
    waist_hip_ratio = waist_width / hip_width if hip_width > 0 else 1.0

    if shoulder_waist_ratio > 1.35:
        body_type = "V-shape (широкі плечі)"
        rec = "Додай тренування на дельти та спину 3–4 рази на тиждень."
    elif waist_hip_ratio > 0.85:
        body_type = "Прямокутник (H-shape)"
        rec = "Збалансуй плечі та талію — силові + кардіо 2–3 рази."
    elif shoulder_waist_ratio > 1.1 and waist_hip_ratio < 0.8:
        body_type = "Пісочний годинник (X-shape)"
        rec = "Талія вузька — акцентуй на ній (кор, об'єм)."
    else:
        body_type = "Яблуко або груша (O/A-shape)"
        rec = "Фокус на ноги та спину, зменш жир на талії кардіо."

    return {
        "shoulder_waist_ratio": round(shoulder_waist_ratio, 2),
        "waist_hip_ratio": round(waist_hip_ratio, 2),
        "body_type": body_type,
        "recommendation": rec,
        "error": None
    }

# Тест (якщо запускаєш файл напряму)
if __name__ == "__main__":
    # Заміни на реальний шлях до свого фото
    photo = r"C:\FitAI_django\media\IMG20230924232928.jpg"
    result = analyze_body_proportions(photo)
    print(result)