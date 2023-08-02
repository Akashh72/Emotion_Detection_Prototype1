# Generated by Django 4.2.3 on 2023-08-02 12:25

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Emotion',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.FileField(upload_to='emotion_images/')),
                ('detected_emotion', models.CharField(max_length=255)),
            ],
        ),
    ]
