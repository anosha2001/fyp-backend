from fastapi import FastAPI, WebSocket
import cv2
import numpy as np
import os

app = FastAPI()

output_dir = "saved_frames"
os.makedirs(output_dir, exist_ok=True)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    frame_count = 0

    while True:
        try:
            # Receive image bytes from the frontend
            data = await websocket.receive_bytes()

            # Convert to OpenCV format
            np_arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                print("Error decoding frame")
                continue  # Skip this iteration if frame is None

            filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved frame: {filename}")

            frame_count += 1

        except Exception as e:
            print(f"WebSocket error: {e}")
            break

    await websocket.close()
