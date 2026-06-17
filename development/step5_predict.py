import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model


IMAGE_SIZE = 64
MODEL_PATH = "models/parking_model.h5"


print("Loading trained model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")


def find_any_image(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith('.jpg'):
            return os.path.join(folder_path, file)
    return None


def safe_imread(image_path):
    
    with open(image_path, 'rb') as f:
        raw_data = np.frombuffer(f.read(), dtype=np.uint8)


    img = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
    return img


def preprocess_image(image_path):

    img = safe_imread(image_path)

    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None


    img_resized    = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))


    img_normalized = img_resized / 255.0

    
    img_batch      = np.expand_dims(img_normalized, axis=0)

    return img, img_batch


def predict_slot(image_path, label_name=""):
    print(f"\nImage      : {os.path.basename(image_path)}")

    
    result = preprocess_image(image_path)


    if result is None:
        print("Skipping this image.")
        return

    original_img, img_batch = result


    prediction = model.predict(img_batch, verbose=0)
    confidence = prediction[0][0]


    if confidence < 0.5:
        label          = "EMPTY"
        color          = (0, 255, 0)        # Green
        confidence_pct = (1 - confidence) * 100
    else:
        label          = "OCCUPIED"
        color          = (0, 0, 255)        # Red
        confidence_pct = confidence * 100

    print(f"Prediction  : {label}")
    print(f"Confidence  : {confidence_pct:.2f}%")
    print(f"Raw output  : {confidence:.4f}")


    display_img = cv2.resize(original_img, (300, 300))


    cv2.putText(display_img, label,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2, color, 3)

    
    cv2.putText(display_img, f"Confidence: {confidence_pct:.1f}%",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2)


    cv2.rectangle(display_img, (0, 0),
                  (299, 299), color, 5)


    cv2.imshow(f"{label_name} → Prediction: {label}", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return label, confidence_pct



empty_folder    = "dataset/PKLotSegmented/PUC/Sunny/2012-09-11/Empty"
occupied_folder = "dataset/PKLotSegmented/PUC/Sunny/2012-09-11/Occupied"


empty_image    = find_any_image(empty_folder)
occupied_image = find_any_image(occupied_folder)

print("\n" + "="*40)
print("Testing on EMPTY slot image...")
print("="*40)

if empty_image:
    predict_slot(empty_image, "EMPTY SLOT TEST")
else:
    print("No empty image found! Check folder path.")

print("\n" + "="*40)
print("Testing on OCCUPIED slot image...")
print("="*40)

if occupied_image:
    predict_slot(occupied_image, "OCCUPIED SLOT TEST")
else:
    print("No occupied image found! Check folder path.")

print("\nStep 5 Complete!")