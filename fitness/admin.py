from django.contrib import admin
from .models import UserData, TrainingPlan  # якщо назви моделей інші — поправ

admin.site.register(UserData)
admin.site.register(TrainingPlan)