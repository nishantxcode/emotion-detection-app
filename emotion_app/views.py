from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from deepface import DeepFace
from PIL import Image
import numpy as np
# import tempfile

def index(request):
    emotion = None

    if request.method == 'POST' and 'image' in request.FILES:
        image_file = request.FILES['image']
        img = Image.open(image_file).convert('RGB')
        img_array = np.array(img)

        try:
            result = DeepFace.analyze(img_array, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion'] if isinstance(result, list) else result['dominant_emotion']
        except Exception as e:
            emotion = "Error"

        # 👇 If it's an AJAX call (like from fetch in webcam mode), return JSON
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            return JsonResponse({'emotion': emotion})
        else:
            # 👇 Otherwise render the page with result
            return render(request, 'index.html', {'emotion': emotion})

    return render(request, 'index.html')



def ping_view(request):
    return HttpResponse("✅ App is alive")