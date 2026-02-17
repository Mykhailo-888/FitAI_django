from django.db import models

class UserSession(models.Model):
    session_key = models.CharField(max_length=40, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Сесія {self.session_key[:8]}"

class UserData(models.Model):
    session = models.ForeignKey(UserSession, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)

    вік = models.FloatField(null=True, blank=True)
    зріст = models.FloatField(null=True, blank=True)
    вага = models.FloatField(null=True, blank=True)
    обхват_талії = models.FloatField(null=True, blank=True)
    емоційний_стрес = models.FloatField(null=True, blank=True)
    алкоголь = models.FloatField(null=True, blank=True)
    калораж = models.FloatField(null=True, blank=True)
    віджимання_max = models.FloatField(null=True, blank=True)
    підтягування_max = models.FloatField(null=True, blank=True)
    біг_1км = models.FloatField(null=True, blank=True)
    біг_100м = models.FloatField(null=True, blank=True)
    тест_купера = models.FloatField(null=True, blank=True)
    бурпі_3хв = models.FloatField(null=True, blank=True)
    віджимання_1хв = models.FloatField(null=True, blank=True)
    сон = models.FloatField(null=True, blank=True)
    пульс_спокою = models.FloatField(null=True, blank=True)
    тиск_верхній = models.FloatField(null=True, blank=True)
    мітохондрії = models.FloatField(null=True, blank=True)
    тестостерон = models.FloatField(null=True, blank=True)
    кортизол = models.FloatField(null=True, blank=True)
    гемоглобін = models.FloatField(null=True, blank=True)
    срб = models.FloatField(null=True, blank=True)

    прогноз = models.JSONField(null=True, blank=True)  # [0.73, ...]

    def __str__(self):
        return f"Дані від {self.timestamp}"

class TrainingPlan(models.Model):
    user_data = models.OneToOneField(UserData, on_delete=models.CASCADE)
    план = models.JSONField()  # словник з планом
    створено = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"План для даних {self.user_data.id}"