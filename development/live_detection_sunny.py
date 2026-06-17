import cv2
import json
import numpy as np
from tensorflow.keras.models import load_model

# ---- Settings ----
IMAGE_SIZE  = 64
MODEL_PATH  = "models/parking_model.h5"
SLOTS_PATH  = "slots.json"
VIDEO_PATH  = "outputs/annotated_images/raw_parking_video_slow.mp4"

# ---- Step 1: Load Model ----
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded!")

# ---- Step 2: Load Slot Coordinates ----
print("Loading slot coordinates...")
with open(SLOTS_PATH, "r") as f:
    slots = json.load(f)
print(f"Total slots: {len(slots)}")

# ---- Step 3: Extract Slot From Frame ----
def extract_slot(image, contour_points):
    pts        = np.array(contour_points, dtype=np.float32)
    x, y, w, h = cv2.boundingRect(pts.astype(np.int32))

    padding = 2
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = w + padding * 2
    h = h + padding * 2

    slot_crop = image[y:y+h, x:x+w]
    return slot_crop

# ---- Step 4: Process Frame With Batch Prediction ----
def process_frame(image):
    output_image   = image.copy()
    empty_count    = 0
    occupied_count = 0

    # Collect all slot crops first
    slot_crops  = []
    valid_slots = []

    for slot in slots:
        contour_points = slot['contour']
        slot_crop      = extract_slot(image, contour_points)

        # Skip if too small
        if slot_crop.size == 0 or \
           slot_crop.shape[0] < 5 or \
           slot_crop.shape[1] < 5:
            continue

        # Resize and normalize
        slot_resized    = cv2.resize(slot_crop, (IMAGE_SIZE, IMAGE_SIZE))
        slot_normalized = slot_resized / 255.0

        slot_crops.append(slot_normalized)
        valid_slots.append(slot)

    # ---- Batch Predict All Slots At Once ----
    if len(slot_crops) == 0:
        return output_image, 0, 0

    # Stack all crops into one batch
    batch = np.array(slot_crops)          # shape: (100, 64, 64, 3)

    # One prediction call for ALL slots!
    predictions = model.predict(batch, verbose=0)

    # ---- Draw Results ----
    for i, slot in enumerate(valid_slots):
        confidence     = predictions[i][0]
        contour_points = slot['contour']

        if confidence < 0.5:
            color = (0, 255, 0)          # Green → Empty
            empty_count += 1
        else:
            color = (0, 0, 255)          # Red → Occupied
            occupied_count += 1

        # Draw slot boundary
        pts = np.array(contour_points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(output_image, [pts],
                      isClosed=True, color=color, thickness=2)

    # Draw counter background
    cv2.rectangle(output_image,
                  (10, 10), (320, 110),
                  (0, 0, 0), -1)

    # Draw counts
    cv2.putText(output_image,
                f"Empty    : {empty_count}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 0), 2)

    cv2.putText(output_image,
                f"Occupied : {occupied_count}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 0, 255), 2)

    return output_image, empty_count, occupied_count

# ---- Step 5: Open Raw Video ----
print(f"\nOpening video: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Could not open video!")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps          = int(cap.get(cv2.CAP_PROP_FPS))
width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Total frames : {total_frames}")
print(f"FPS          : {fps}")

# ---- Step 6: Setup Output Video Writer ----
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter(
    "outputs/annotated_images/live_output_video.mp4",
    fourcc, fps, (width, height)
)

print("\n" + "="*40)
print("LIVE DETECTION STARTED!")
print("Press Q to quit anytime")
print("="*40 + "\n")

frame_count = 0

# ---- Step 7: Process Video Frame by Frame ----
while True:
    ret, frame = cap.read()

    if not ret:
        print("\nVideo processing complete!")
        break

    frame_count += 1

    # Process frame
    output_frame, empty, occupied = process_frame(frame)

    # Add frame number
    cv2.putText(output_frame,
                f"Frame: {frame_count}/{total_frames}",
                (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1)

    # Write to output video
    out.write(output_frame)

    # Show live
    cv2.imshow("Smart Parking - Live Detection", output_frame)

    # Print progress
    progress = frame_count / total_frames * 100
    print(f"Frame {frame_count}/{total_frames} "
          f"({progress:.1f}%) → "
          f"Empty: {empty} | Occupied: {occupied}")

    # Press Q to quit
    if cv2.waitKey(500) & 0xFF == ord('q'):
        print("\nStopped by user!")
        break

# ---- Step 8: Cleanup ----
cap.release()
out.release()
cv2.destroyAllWindows()

print("\n" + "="*40)
print("DETECTION COMPLETE!")
print("="*40)
print(f"Frames processed : {frame_count}")
print(f"Output saved to  : outputs/annotated_images/live_output_video.mp4")
print("="*40)