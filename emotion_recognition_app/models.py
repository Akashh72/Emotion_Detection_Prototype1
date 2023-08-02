# models.py
from django.db import models

class Emotion(models.Model):
    image = models.FileField(upload_to='emotion_images/')
    detected_emotion = models.CharField(max_length=255)
