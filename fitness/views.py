from django.shortcuts import render, redirect
from django.urls import reverse
import numpy as np
from ml.fit_model_core import FitnessNeuralNet
from ml.training_optimizer import weekly_training_plan_optimizer

def onboarding(request):
    questions = [
        ("Вік", "років", "int", True),
        ("Зріст", "см", "float", True),
        ("Вага", "кг", "float", True),
        ("Завантажити фото для аналізу пропорцій", "фото", "photo", False),
        ("Обхват талії", "см", "float", False),
        ("Емоційний стрес", "1-10", "int", False),
        ("Алкоголь (одиниць/тиждень)", "одиниць", "int", False),
        ("Калораж (ккал/день)", "ккал", "float", False),
        ("Віджимання max", "разів", "int", False),
        ("Підтягування max", "разів", "int", False),
        ("Біг 1 км", "хв", "float", False),
        ("Біг 100 м", "сек", "float", False),
        ("Тест Купера", "хв", "float", False),
        ("Бурпі за 3 хв", "разів", "int", False),
        ("Віджимання за 1 хв", "разів", "int", False),
        ("Сон", "год/добу", "float", False),
        ("HRV", "одиниць", "float", False),
        ("Пульс спокою", "BPM", "int", False),
        ("Тиск верхній", "мм рт.ст.", "int", False),
        ("Мітохондрії (заглушка)", "одиниць", "float", False),
        ("Тестостерон", "нг/дл", "float", False),
        ("Кортизол", "мкг/дл", "float", False),
        ("Гемоглобін", "г/дл", "float", False),
        ("СРБ", "мг/л", "float", False),
    ]

    current_index = int(request.GET.get('q', 0))

    if current_index >= len(questions):
        return redirect(reverse('onboarding'))

    error = None
    previous_value = None

    if request.method == 'POST':
        request.session['data'] = request.session.get('data', {})
        question = questions[current_index][0]
        value = request.POST.get(question)

        # Валідація
        validators = {
            "Вік": lambda v: 18 <= v <= 80,
            "Зріст": lambda v: 140 <= v <= 220,
            "Вага": lambda v: 40 <= v <= 200,
            "Обхват талії": lambda v: 50 <= v <= 150,
            "Емоційний стрес": lambda v: 1 <= v <= 10,
            "Алкоголь (одиниць/тиждень)": lambda v: 0 <= v <= 50,
            "Калораж (ккал/день)": lambda v: 1000 <= v <= 6000,
            "Віджимання max": lambda v: 0 <= v <= 100,
            "Підтягування max": lambda v: 0 <= v <= 50,
            "Біг 1 км": lambda v: 3 <= v <= 20,
            "Біг 100 м": lambda v: 10 <= v <= 30,
            "Тест Купера": lambda v: 1 <= v <= 5,
            "Бурпі за 3 хв": lambda v: 10 <= v <= 100,
            "Віджимання за 1 хв": lambda v: 10 <= v <= 100,
            "Сон": lambda v: 3 <= v <= 12,
            "HRV": lambda v: 20 <= v <= 150,
            "Пульс спокою": lambda v: 40 <= v <= 120,
            "Тиск верхній": lambda v: 80 <= v <= 180,
            "Мітохондрії (заглушка)": lambda v: 0 <= v <= 100,
            "Тестостерон": lambda v: 100 <= v <= 1500,
            "Кортизол": lambda v: 5 <= v <= 25,
            "Гемоглобін": lambda v: 10 <= v <= 18,
            "СРБ": lambda v: v >= 0,  # тільки невід'ємне
        }

        if questions[current_index][2] != "photo":
            if not value and questions[current_index][3]:
                error = f"{question} — обов’язкове поле"
            else:
                try:
                    value = float(value) if questions[current_index][2] == "float" else int(value)
                    if question in validators and not validators[question](value):
                        error = f"{question} — значення поза межами"
                except (ValueError, TypeError):
                    error = f"{question} — має бути число"

        if error:
            context = {
                'question': questions[current_index][0],
                'unit': questions[current_index][1],
                'dtype': questions[current_index][2],
                'required': questions[current_index][3],
                'current_index': current_index,
                'total_questions': len(questions),
                'is_first': current_index == 0,
                'error': error,
                'previous_value': value,
            }
            return render(request, 'onboarding.html', context)

        # Збереження
        request.session['data'][question] = value

        next_index = current_index + 1
        if next_index >= len(questions):
            data = request.session.get('data', {})

            model = FitnessNeuralNet()
            model.load_model("models/trained_fitness_model.pkl")

            kaggle_features = ["Вік", "Зріст", "Вага", "Обхват талії", "Пульс спокою", "Сон"]

            data_values = [float(data.get(f, 0)) for f in kaggle_features]
            data_array = np.array(data_values).reshape(1, -1)

            if model.mean_X is not None and model.std_X is not None:
                data_array = (data_array - model.mean_X) / model.std_X
            data_array = np.clip(data_array, -5, 5)

            pred = model.predict(data_array)

            weekly_plan = weekly_training_plan_optimizer(
                current_weight=float(data.get("Вага", 94.0)),
                target_weight=88.0,
                weeks_left=12,
                current_hrv=float(data.get("HRV", 50.0)),
                sleep_hours=float(data.get("Сон", 7.0)),
                alcohol_units=float(data.get("Алкоголь (одиниць/тиждень)", 0)),
                training_load_avg=float(data.get("Калораж (ккал/день)", 500)),
                age=float(data.get("Вік", 40)),
                systolic_bp=float(data.get("Тиск верхній", 130)),
                resting_bpm=float(data.get("Пульс спокою", 70))
            )

            weekly_plan_formatted = {
                key.replace("_", " ").title(): round(value, 2) if isinstance(value, (float, np.float64)) else value
                for key, value in weekly_plan.items()
            }

            # Обробка високого СРБ
            srb = float(data.get("СРБ", 1.0))
            high_crp_warning = None
            training_restriction = None

            if srb > 10:
                high_crp_warning = f"УВАГА! СРБ = {srb} мг/л — це дуже високий рівень! Можлива гостра інфекція, травма або серйозна проблема. Тренування **категорично протипоказані**. Негайно звернися до лікаря!"
                training_restriction = "Тренування заборонені до консультації з лікарем"
            elif srb > 5:
                high_crp_warning = f"СРБ = {srb} мг/л — підвищений рівень запалення. Зменш навантаження, додай протизапальні заходи (омега-3 2–3 г/день, куркумін, відпочинок) і проконсультуйся з лікарем."
                training_restriction = "Рекомендується тільки легке навантаження або відновлення"

            # Персоналізовані рекомендації
            recommendations = []

            sleep = float(data.get("Сон", 7.0))
            stress = float(data.get("Емоційний стрес", 5))
            alcohol = float(data.get("Алкоголь (одиниць/тиждень)", 0))

            if sleep < 7:
                recommendations.append(f"Сон тільки {sleep} годин — збільш до 7.5–8.5, це критично для відновлення.")
            elif sleep > 9:
                recommendations.append(f"Сон {sleep} годин — добре, але стеж за якістю (темрява, без телефону).")
            else:
                recommendations.append(f"Сон {sleep} годин — в нормі, тримай так.")

            if stress >= 7:
                recommendations.append(f"Стрес високий ({stress}/10) — додай 10–15 хв медитації щодня.")
            elif stress <= 3:
                recommendations.append(f"Стрес низький ({stress}/10) — можеш збільшити навантаження.")

            if alcohol > 10:
                recommendations.append(f"Алкоголь {alcohol} одиниць/тиждень — зменш до 5, це гальмує відновлення.")

            if srb > 3:
                recommendations.append(f"СРБ {srb} мг/л — підвищений, додай омега-3 (2–3 г/день), куркумін, перевір з лікарем.")

            recommendations_text = "\n".join([f"• {rec}" for rec in recommendations]) or "Усе в нормі — тримай темп!"

            return render(request, 'results.html', {
                'prediction': np.round(pred, 2).tolist(),
                'weekly_plan': weekly_plan_formatted,
                'high_crp_warning': high_crp_warning,
                'training_restriction': training_restriction,
                'recommendations': recommendations_text,
            })

        return redirect(reverse('onboarding') + f'?q={next_index}')

    question, unit, dtype, required = questions[current_index]
    context = {
        'question': question,
        'unit': unit,
        'dtype': dtype,
        'required': required,
        'current_index': current_index,
        'total_questions': len(questions),
        'is_first': current_index == 0,
    }

    return render(request, 'onboarding.html', context)