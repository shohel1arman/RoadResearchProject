from ultralytics import YOLO
import cv2
import os
from dotenv import load_dotenv
import ollama
import base64
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView

load_dotenv()
model_path = os.getenv("MODEL_PATH2")
model = YOLO(model_path)

def generate_road_damage_report(image_path: str):
    image = cv2.imread(image_path)
    results = model(image_path)

    damage_counts = {"Crack": 0, "Pothole": 0, "Surface Erosion": 0}
    for cls_id in results[0].boxes.cls:
        label = model.names[int(cls_id)]
        if label in damage_counts:
            damage_counts[label] += 1

    pothole_count = damage_counts["Pothole"]
    crack_count = damage_counts["Crack"]
    erosion_count = damage_counts["Surface Erosion"]

    pothole_score = pothole_count * 3
    crack_score = crack_count * 2
    erosion_score = erosion_count * 4
    score = pothole_score + crack_score + erosion_score
    if pothole_count >= 6 and erosion_count >= 3:
        condition = "Critical"
        explanation = "Severe pothole and surface erosion detected."
    elif pothole_count >= 6:
        condition = "Very Poor"
        explanation = "High number of potholes detected."
    elif pothole_count >= 3 and erosion_count >= 2:
        condition = "Poor"
        explanation = "Potholes and erosion indicate poor quality."
    elif erosion_count >= 4:
        condition = "Poor"
        explanation = "Significant surface erosion."
    elif pothole_count >= 3 or crack_count >= 6:
        condition = "Moderate"
        explanation = "Moderate road damage."
    elif crack_count >= 3:
        condition = "Fair"
        explanation = "Cracks found, not critical."
    else:
        condition = "Good"
        explanation = "Minor damage."
    
    print(score)
    print(condition)
    print(explanation)

    # Build prompt
    has_pothole = pothole_count > 0
    has_crack = crack_count > 0
    has_erosion = erosion_count > 0

    detected_types = []
    if has_pothole:
        detected_types.append(f"- Potholes: {pothole_count}")
    if has_crack:
        detected_types.append(f"- Cracks: {crack_count}")
    if has_erosion:
        detected_types.append(f"- Surface Erosion: {erosion_count}")

    if detected_types:
        prompt = f"""
        Detected road damage:
        {chr(10).join(detected_types)}
        Overall road condition is {condition}.

        Write a short, professional report summarizing the road damage condition for municipal assessment. Use clear, non-technical language.
        """
    else:
        prompt = f"""
        Detected road damage:
        No major damage detected.
        Overall road condition is {condition}.

        Write a short, professional report summarizing the road damage condition for municipal assessment. Use clear, non-technical language.
        """

    response = ollama.chat(
        model='llama3.2',
        messages=[{'role': 'user', 'content': prompt}]
    )
    return prompt.strip(), response['message']['content'].strip()



def generate_stream():
    stream_url = "http://192.168.0.102:4747/video"
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Run YOLOv8 inference
        results = model(frame)
        annotated_frame = results[0].plot()

        # Convert to JPEG
        ret, jpeg = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue

        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()