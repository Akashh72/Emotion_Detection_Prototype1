# views.py
from django.shortcuts import render
from .models import Emotion
from .utils import perform_emotion_recognition
from PIL import Image, ImageDraw, ImageFont
import io
import base64

def emotion_recognition(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        
        emotions_detected = perform_emotion_recognition(image)

        detected_emotion_str = ", ".join([emotion for emotion, _ in emotions_detected])

        emotion_instance = Emotion(image=image, detected_emotion=detected_emotion_str)
        emotion_instance.save()

        img = Image.open(image)
        draw = ImageDraw.Draw(img)

        font = ImageFont.truetype("emotion_recognition_app/fonts/arial.ttf", size=40)

        for emotion, (x, y, w, h) in emotions_detected:
            draw.rectangle([x, y, x + w, y + h], outline="white")
            draw.text((x, y - 20), emotion, fill="red", font=font)

        buffered = io.BytesIO()
        img.save(buffered, format=img.format)
        img_str = base64.b64encode(buffered.getvalue()).decode()

        emotions_detected_with_images = []
        for emotion in emotions_detected:
            emotions_detected_with_images.append((emotion, emotion_instance.image.url))

        # Delete the image record from the database
        emotion_instance.delete()

        return render(request, 'emotion_recognition.html', {'emotions_detected': emotions_detected_with_images, 'detected_emotion': detected_emotion_str, 'main_image': img_str})
    
    return render(request, 'emotion_recognition.html')
