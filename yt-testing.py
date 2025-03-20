#%%
import cv2
import torch
from transformers import AutoProcessor, AutoModelForImageClassification
from collections import Counter
import os

# Load the model and processor globally
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

def extract_frames(video_path, every_nth=5, frame_size=(224, 224)):
    """
    Extracts every nth frame from a video and resizes it to the given size.

    :param video_path: Path to the video file.
    :param every_nth: Interval for frame extraction.
    :param frame_size: Tuple for resizing frames (width, height).
    :return: List of resized frames.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % every_nth == 0:
            frame = cv2.resize(frame, frame_size)
            frames.append(frame)
        count += 1

    cap.release()
    # print(frames.count())
    return frames

def predict_video_labels(video_path):
    """
    Processes a video file, extracts frames, predicts labels, and returns the top 2 labels.

    :param video_path: Path to the video file.
    :return: Top 2 most occurring labels with counts.
    """
    # Extract frames
    frames = extract_frames(video_path)

    # Preprocess frames for the model
    inputs = []
    for frame in frames:
        input_tensor = processor(images=frame, return_tensors="pt").pixel_values
        inputs.append(input_tensor)

    # Concatenate all frames into a single tensor
    inputs = torch.cat(inputs)

    # Pass frames through the model
    with torch.no_grad():
        outputs = model(pixel_values=inputs)

    # Get predictions
    predictions = torch.argmax(outputs.logits, dim=-1)
    print("Predictions for :",video_path, predictions)
    # Count occurrences of each ID
    predicted_ids = predictions.tolist()
    id_counts = Counter(predicted_ids)

    # Get the top 2 most occurring IDs
    top_2_ids = id_counts.most_common(2)

    # Map the IDs to labels
    top_2_labels = [(id_to_label[id_], count) for id_, count in top_2_ids]

    return top_2_labels

def main(video_path):
    """
    Main function to process the video and print the top 2 labels.

    :param video_path: Path to the video file.
    """
    top_2_labels = predict_video_labels(video_path)
    print("Top 2 most occurring labels with counts:", top_2_labels)

# Example usage
if __name__ == "__main__":
    video_path = "C:/Users/anosha/Downloads/road-accident1.mp4"
    main(video_path)
    video_path2 = "C:/Users/anosha/Downloads/robbery1.mp4"
    main(video_path2)
    video_path3 = "C:/Users/anosha/Downloads/explosion1.mp4"
    main(video_path3)
