import json
import os
from datetime import datetime
from collections import Counter
from typing import Dict

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket
from transformers import AutoProcessor, AutoModelForImageClassification
from ultralytics import YOLO

import firebase_admin
from firebase_admin import credentials, firestore

app = FastAPI()

# === Load ViT Model ===
VIT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "vit")
vit_model = AutoModelForImageClassification.from_pretrained(VIT_MODEL_DIR)
vit_processor = AutoProcessor.from_pretrained(VIT_MODEL_DIR, use_fast=True)

# === Load YOLOv8 Model ===
YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "yolov8", "best.pt")
yolo_model = YOLO(YOLO_MODEL_PATH)

# === Label mappings ===
id_to_label = {
    0: 'Abuse', 1: 'Arrest', 2: 'Arson', 3: 'Assault', 4: 'Burglary',
    5: 'Explosion', 6: 'Fighting', 7: 'Normal_Videos', 8: 'RoadAccidents',
    9: 'Robbery', 10: 'Shooting', 11: 'Shoplifting', 12: 'Stealing', 13: 'Vandalism'
}

weapon_labels = ['pistol', 'smartphone', 'knife', 'monedero', 'billete', 'tarjeta']
weapon_classes_to_alert = {'pistol', 'knife'}

frame_predictions_dict: Dict[str, list] = {}
last_sent_alerts: Dict[str, str] = {}  # Tracks last alert per camera


# === Firebase Initialization ===
firebase_path = os.path.join(os.path.dirname(__file__), "firebase_config.json")
cred = credentials.Certificate(firebase_path)
firebase_admin.initialize_app(cred)
db = firestore.client()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    camera_id = None

    while True:
        try:
            message = await websocket.receive()

            if "text" in message:
                try:
                    camera_info = json.loads(message["text"])
                    camera_id = str(camera_info.get("camera_id", "unknown"))
                    if camera_id not in frame_predictions_dict:
                        frame_predictions_dict[camera_id] = []
                    if camera_id not in last_sent_alerts:
                        last_sent_alerts[camera_id] = None
                except Exception as e:
                    print("Failed to parse camera ID JSON:", e)
                    continue

            elif "bytes" in message:
                if camera_id is None:
                    print("Camera ID not received before frame")
                    continue

                # Decode frame
                np_arr = np.frombuffer(message["bytes"], np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is None:
                    print("Error decoding frame")
                    continue

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                frame_resized = cv2.resize(frame, (224, 224))

                # === ViT Classification ===
                input_tensor = vit_processor(images=frame_resized, return_tensors="pt").pixel_values
                with torch.no_grad():
                    output = vit_model(pixel_values=input_tensor)
                predicted_id = torch.argmax(output.logits, dim=-1).item()
                predicted_label = id_to_label[predicted_id]

                frame_predictions_dict[camera_id].append(predicted_label)
                if len(frame_predictions_dict[camera_id]) > 10:
                    frame_predictions_dict[camera_id].pop(0)

                label_counts = Counter(frame_predictions_dict[camera_id])
                most_common_label, count = label_counts.most_common(1)[0]
                activity_alert = None

                # === Save Anomalous Frame & Store in Firebase ===
                if count >= 7 and most_common_label != 'Normal_Videos':
                    activity_alert = f"Alert: {most_common_label}"

                    # Save frame locally
                    timestamp_filename = datetime.now().strftime("%Y%m%d_%H%M%S")
                    folder_path = os.path.join("saved_frames", camera_id)
                    os.makedirs(folder_path, exist_ok=True)
                    frame_filename = f"{timestamp_filename}.jpg"
                    frame_path = os.path.join(folder_path, frame_filename)
                    cv2.imwrite(frame_path, frame)

                    print(f"[{camera_id}] Anomalous frame saved to: {os.path.abspath(frame_path)}")

                    # Save metadata to Firebase
                    data = {
                        "camera_id": camera_id,
                        "anomaly": most_common_label,
                        "timestamp": timestamp,
                        "frame_path": os.path.abspath(frame_path)
                    }
                    db.collection("anomalous_frames").add(data)

                # === YOLO Detection ===
                yolo_results = yolo_model(frame_resized, verbose=False)
                detected_objects = []
                weapon_alert = None

                for result in yolo_results:
                    if result.boxes is not None:
                        for cls_id in result.boxes.cls:
                            label = weapon_labels[int(cls_id)]
                            detected_objects.append(label)

                weapon_types = set(detected_objects)
                threatening_weapons = weapon_types.intersection(weapon_classes_to_alert)

                if threatening_weapons:
                    weapon_alert = f"Weapons detected: {', '.join(threatening_weapons)}"

                # === Combine Alerts ===
                combined_alert = None
                if activity_alert and weapon_alert:
                    combined_alert = f"{activity_alert} | {weapon_alert}"
                elif activity_alert:
                    combined_alert = activity_alert
                elif weapon_alert:
                    combined_alert = weapon_alert

                if combined_alert != last_sent_alerts.get(camera_id):
                    last_sent_alerts[camera_id] = combined_alert

                    response = {
                        "timestamp": timestamp,
                        "camera_id": camera_id,
                        "predicted_activity": predicted_label,
                        "activity_trend": most_common_label,
                        "detected_objects": detected_objects,
                        "alert": combined_alert
                    }

                # print(f"[{camera_id}] Activity: {predicted_label}, Objects: {detected_objects}, Alert: {combined_alert}")
                    print(f"[{camera_id}] Alert: {combined_alert}")
                    await websocket.send_json(response)

        except Exception as e:
            print(f"WebSocket error: {e}")
            break

    await websocket.close()
