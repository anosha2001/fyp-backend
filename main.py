import json
import os
from datetime import datetime
from collections import Counter
from typing import Dict

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, Query
from fastapi.responses import JSONResponse
from transformers import AutoProcessor, AutoModelForImageClassification
from ultralytics import YOLO

from db import insert_anomaly, fetch_alerts

app = FastAPI()

# === Load Models ===
VIT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "vit")
vit_model = AutoModelForImageClassification.from_pretrained(VIT_MODEL_DIR)
vit_processor = AutoProcessor.from_pretrained(VIT_MODEL_DIR, use_fast=True)

YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "yolov8", "best.pt")
yolo_model = YOLO(YOLO_MODEL_PATH)

# === Constants ===
id_to_label = {
    0: 'Abuse', 1: 'Arrest', 2: 'Arson', 3: 'Assault', 4: 'Burglary',
    5: 'Explosion', 6: 'Fighting', 7: 'Normal_Videos', 8: 'RoadAccidents',
    9: 'Robbery', 10: 'Shooting', 11: 'Shoplifting', 12: 'Stealing', 13: 'Vandalism'
}
weapon_labels = ['pistol', 'smartphone', 'knife', 'monedero', 'billete', 'tarjeta']
weapon_classes_to_alert = {'pistol', 'knife'}

# === State ===
frame_predictions_dict: Dict[str, list] = {}
last_sent_alerts: Dict[str, str] = {}

# === Ensure image directory exists ===
IMAGE_DIR = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

# === WebSocket Endpoint ===
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
                    frame_predictions_dict.setdefault(camera_id, [])
                    last_sent_alerts.setdefault(camera_id, "empty")
                except Exception as e:
                    print("Failed to parse camera ID JSON:", e)
                    continue

            elif "bytes" in message:
                if camera_id is None:
                    continue

                np_arr = np.frombuffer(message["bytes"], np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                frame_resized = cv2.resize(frame, (224, 224))

                # === ViT ===
                input_tensor = vit_processor(images=frame_resized, return_tensors="pt").pixel_values
                with torch.no_grad():
                    output = vit_model(pixel_values=input_tensor)
                predicted_label = id_to_label[torch.argmax(output.logits, dim=-1).item()]

                frame_predictions_dict[camera_id].append(predicted_label)
                if len(frame_predictions_dict[camera_id]) > 10:
                    frame_predictions_dict[camera_id].pop(0)

                label_counts = Counter(frame_predictions_dict[camera_id])
                most_common_label, count = label_counts.most_common(1)[0]
                activity_alert = None

                if count >= 7 and most_common_label != 'Normal_Videos':
                    activity_alert = f"Alert: {most_common_label}"

                # === YOLO ===
                yolo_results = yolo_model(frame_resized, verbose=False)
                detected_objects = [
                    weapon_labels[int(cls_id)]
                    for result in yolo_results
                    if result.boxes is not None
                    for cls_id in result.boxes.cls
                ]
                weapon_types = set(detected_objects)
                threatening_weapons = weapon_types.intersection(weapon_classes_to_alert)
                weapon_alert = f"Weapons detected: {', '.join(threatening_weapons)}" if threatening_weapons else None

                # === Combine Alerts ===
                combined_alert = activity_alert
                if activity_alert and weapon_alert:
                    combined_alert += f" | {weapon_alert}"
                elif weapon_alert:
                    combined_alert = weapon_alert

                if combined_alert != last_sent_alerts[camera_id]:
                    last_sent_alerts[camera_id] = combined_alert

                    # === Save frame ===
                    filename = f"{camera_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    frame_path = os.path.join(IMAGE_DIR, filename)
                    cv2.imwrite(frame_path, frame)

                    # === Save to DB ===
                    anomaly_type = most_common_label if activity_alert else list(threatening_weapons)[0]
                    insert_anomaly(camera_id, timestamp, anomaly_type, frame_path)

                    # === Send Alert ===
                    await websocket.send_json({
                        "timestamp": timestamp,
                        "camera_id": camera_id,
                        "predicted_activity": predicted_label,
                        "activity_trend": most_common_label,
                        "detected_objects": detected_objects,
                        "alert": combined_alert,
                        "frame_path": frame_path
                    })

        except Exception as e:
            print(f"WebSocket error: {e}")
            break

    await websocket.close()

# === Alerts API ===
@app.get("/alerts")
def get_alerts(
    camera_id: str = Query(None),
    anomaly_type: str = Query(None),
    timestamp: str = Query(None)
):
    try:
        records = fetch_alerts(camera_id, anomaly_type, timestamp)
        return JSONResponse([
            {
                "camera_id": r[0],
                "timestamp": r[1].strftime("%Y-%m-%d %H:%M:%S"),
                "anomaly_type": r[2],
                "frame_path": r[3]
            }
            for r in records
        ])
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
