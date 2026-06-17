import cv2
import json
import numpy as np
from tensorflow.keras.models import load_model

IMAGE_SIZE = 64
MODEL_PATH = "models/parking_model.h5"
IMAGE_PATH = "dataset/PKLot/PUCPR/Sunny/2012-09-11/2012-09-11_16_24_53.jpg"
SLOTS_PATH = "slots.json"


print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded!")


def safe_imread(image_path):
    with open(image_path, 'rb') as f:
        raw_data = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
    return img


print("Loading slot coordinates...")
with open(SLOTS_PATH, "r") as f:
    slots = json.load(f)
print(f"Total slots loaded: {len(slots)}")

def extract_slot(image, contour_points):
    
    pts = np.array(contour_points, dtype=np.float32)

    
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
        label          = "Empty"
        color          = (0, 255, 0)        # Green
        confidence_pct = (1 - confidence) * 100
    else:
        label          = "Occupied"
        color          = (0, 0, 255)        # Red
        confidence_pct = confidence * 100

    return label, color, confidence_pct


print("\nLoading parking lot image...")
image = safe_imread(IMAGE_PATH)

if image is None:
    print("Error: Could not load image!")
    exit()

print("Image loaded successfully!")
output_image   = image.copy()

empty_count    = 0
occupied_count = 0
error_count    = 0

print("\nProcessing all slots...")

for slot in slots:
    slot_id        = slot['id']
    contour_points = slot['contour']

    
    slot_crop = extract_slot(image, contour_points)

    if slot_crop.size == 0 or slot_crop.shape[0] < 5 or slot_crop.shape[1] < 5:
        error_count += 1
        continue

    
    label, color, confidence_pct = predict_slot(slot_crop)


    if label == "Empty":
        empty_count += 1
    else:
        occupied_count += 1

    
    pts = np.array(contour_points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(output_image, [pts],
                  isClosed=True, color=color, thickness=2)


    cx = contour_points[0][0]
    cy = contour_points[0][1] - 5
    cv2.putText(output_image, str(slot_id),
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35, color, 1)


cv2.rectangle(output_image, (10, 10), (300, 100), (0, 0, 0), -1)

cv2.putText(output_image, f"Empty    : {empty_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0), 2)


cv2.putText(output_image, f"Occupied : {occupied_count}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 0, 255), 2)


print("\n" + "="*40)
print("RESULTS")
print("="*40)
print(f"Total slots processed : {empty_count + occupied_count}")
print(f"Empty slots           : {empty_count}")
print(f"Occupied slots        : {occupied_count}")
if error_count > 0:
    print(f"Skipped slots         : {error_count}")
print("="*40)


import os
os.makedirs("outputs/annotated_images", exist_ok=True)
output_path = "outputs/annotated_images/full_image_test.jpg"
cv2.imwrite(output_path, output_image)
print(f"\nAnnotated image saved to: {output_path}")


cv2.imshow("Smart Parking - Full Image Test", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\nStep 6 Full Image Test Complete!")
