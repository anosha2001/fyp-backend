import os

from fastapi import FastAPI, WebSocket
import cv2
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForImageClassification
from collections import Counter

app = FastAPI()

# Load model and processor globally
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "vit")

# Load model and processor globally
model = AutoModelForImageClassification.from_pretrained(MODEL_DIR)
processor = AutoProcessor.from_pretrained(MODEL_DIR, use_fast=True)


# Mapping of IDs to Labels
id_to_label = {
    0: 'Abuse', 1: 'Arrest', 2: 'Arson', 3: 'Assault', 4: 'Burglary',
    5: 'Explosion', 6: 'Fighting', 7: 'Normal_Videos', 8: 'RoadAccidents',
    9: 'Robbery', 10: 'Shooting', 11: 'Shoplifting', 12: 'Stealing', 13: 'Vandalism'
}

# Store past predictions for alert generation
frame_predictions = []


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        try:
            # Receive image bytes from the frontend
            data = await websocket.receive_bytes()

            # Convert bytes to OpenCV image
            np_arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                print("Error decoding frame")
                continue

            # Resize frame for model input
            frame_resized = cv2.resize(frame, (224, 224))

            # Preprocess for model
            input_tensor = processor(images=frame_resized, return_tensors="pt").pixel_values

            # Run inference
            with torch.no_grad():
                output = model(pixel_values=input_tensor)

            # Get predicted label ID
            predicted_id = torch.argmax(output.logits, dim=-1).item()
            predicted_label = id_to_label[predicted_id]

            # Store prediction for trend detection
            frame_predictions.append(predicted_label)
            if len(frame_predictions) > 10:  # Keep last 10 predictions
                frame_predictions.pop(0)

            # Count occurrences of labels in recent frames
            label_counts = Counter(frame_predictions)
            most_common_label, count = label_counts.most_common(1)[0]

            # Send alert if a suspicious label appears frequently
            if most_common_label != "Normal_Videos" and count >= 7:  # Threshold for anomaly
                alert_message = f"ALERT: High occurrence of '{most_common_label}' detected!"
                await websocket.send_json({"alert": alert_message})

            print(f"Frame classified as: {predicted_label}")

        except Exception as e:
            print(f"WebSocket error: {e}")
            break

    await websocket.close()
