import cv2
import json
import numpy as np
import os
from tensorflow.keras.models import load_model


IMAGE_SIZE  = 64
MODEL_PATH  = "models/parking_model.h5"
SLOTS_PATH  = "slots.json"
IMAGE_FOLDER = "dataset/PKLot/PUCPR/Sunny/2012-09-15"
OUTPUT_PATH  = "outputs/annotated_images/parking_video.mp4"
FPS          = 5          


print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded!")


print("Loading slot coordinates...")
with open(SLOTS_PATH, "r") as f:
    slots = json.load(f)
print(f"Total slots: {len(slots)}")

def safe_imread(image_path):
    with open(image_path, 'rb') as f:
        raw_data = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
    return img


def extract_slot(image, contour_points):
    pts     = np.array(contour_points, dtype=np.float32)
    x, y, w, h = cv2.boundingRect(pts.astype(np.int32))

    padding = 2
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = w + padding * 2
    h = h + padding * 2

    slot_crop = image[y:y+h, x:x+w]
    return slot_crop


def preprocess_slot(slot_img):
    slot_resized    = cv2.resize(slot_img, (IMAGE_SIZE, IMAGE_SIZE))
    slot_normalized = slot_resized / 255.0
    slot_batch      = np.expand_dims(slot_normalized, axis=0)
    return slot_batch


def predict_slot(slot_img):
    slot_batch = preprocess_slot(slot_img)
    prediction = model.predict(slot_batch, verbose=0)
    confidence = prediction[0][0]

    if confidence < 0.5:
        label = "Empty"
        color = (0, 255, 0)             # Green
    else:
        label = "Occupied"
        color = (0, 0, 255)             # Red

    return label, color


def process_frame(image):
    output_image   = image.copy()
    empty_count    = 0
    occupied_count = 0

    for slot in slots:
        contour_points = slot['contour']

        # Extract slot
        slot_crop = extract_slot(image, contour_points)

        # Skip if too small
        if slot_crop.size == 0 or slot_crop.shape[0] < 5 or slot_crop.shape[1] < 5:
            continue

        # Predict
        label, color = predict_slot(slot_crop)

        # Count
        if label == "Empty":
            empty_count += 1
        else:
            occupied_count += 1

        # Draw slot boundary
        pts = np.array(contour_points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(output_image, [pts],
                      isClosed=True, color=color, thickness=2)


    cv2.rectangle(output_image, (10, 10), (320, 110), (0, 0, 0), -1)

    
    cv2.putText(output_image, f"Empty    : {empty_count}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 0), 2)

    cv2.putText(output_image, f"Occupied : {occupied_count}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 0, 255), 2)

    return output_image, empty_count, occupied_count


print("\nCollecting images...")
image_files = sorted([
    f for f in os.listdir(IMAGE_FOLDER)
    if f.endswith('.jpg')
])
print(f"Total images found: {len(image_files)}")



first_image = safe_imread(os.path.join(IMAGE_FOLDER, image_files[0]))
height, width = first_image.shape[:2]

os.makedirs("outputs/annotated_images", exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (width, height))

print(f"Video dimensions : {width}x{height}")
print(f"FPS              : {FPS}")
print(f"Output path      : {OUTPUT_PATH}")


print("\nProcessing frames...")
print("This will take a few minutes, please wait...\n")

for i, image_file in enumerate(image_files):
    image_path = os.path.join(IMAGE_FOLDER, image_file)

    
    image = safe_imread(image_path)

    if image is None:
        print(f"Skipping: {image_file}")
        continue

    output_frame, empty, occupied = process_frame(image)


    video_writer.write(output_frame)


    progress = (i + 1) / len(image_files) * 100
    print(f"Progress: {i+1}/{len(image_files)} "
          f"({progress:.1f}%) → "
          f"Empty: {empty} | Occupied: {occupied}")


video_writer.release()

print("\n" + "="*40)
print("VIDEO CREATED SUCCESSFULLY!")
print("="*40)
print(f"Total frames    : {len(image_files)}")
print(f"Video saved to  : {OUTPUT_PATH}")
print(f"Video length    : ~{len(image_files) // FPS} seconds")
print("="*40)
print("\nStep 6 Complete!")