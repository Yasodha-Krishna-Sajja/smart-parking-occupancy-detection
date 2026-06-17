import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split


IMAGE_SIZE    = 64          
SAMPLE_SIZE   = 500         
BASE_PATH     = "dataset/PKLotSegmented"


def collect_image_paths(base_path, sample_size):
    empty_paths    = []
    occupied_paths = []


    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.jpg'):
                full_path = os.path.join(root, file)
                if 'Empty' in root:
                    empty_paths.append(full_path)
                elif 'Occupied' in root:
                    occupied_paths.append(full_path)


    np.random.seed(42)      
    empty_paths    = np.random.choice(empty_paths,    sample_size, replace=False)
    occupied_paths = np.random.choice(occupied_paths, sample_size, replace=False)

    print(f"Empty paths collected    : {len(empty_paths)}")
    print(f"Occupied paths collected : {len(occupied_paths)}")

    return empty_paths, occupied_paths


def load_and_preprocess(image_paths, label):
    images = []
    labels = []

    for path in image_paths:
    
        img = cv2.imread(path)

        if img is None:
            continue

        
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

       
        img = img / 255.0

        images.append(img)
        labels.append(label)

    return images, labels


print("Starting preprocessing...")
print(f"Sample size : {SAMPLE_SIZE} empty + {SAMPLE_SIZE} occupied")
print(f"Image size  : {IMAGE_SIZE}x{IMAGE_SIZE}")
print("-" * 40)


empty_paths, occupied_paths = collect_image_paths(BASE_PATH, SAMPLE_SIZE)


print("\nLoading empty images...")
empty_images, empty_labels       = load_and_preprocess(empty_paths, label=0)
print(f"Loaded {len(empty_images)} empty images")

print("\nLoading occupied images...")
occupied_images, occupied_labels = load_and_preprocess(occupied_paths, label=1)
print(f"Loaded {len(occupied_images)} occupied images")


all_images = np.array(empty_images + occupied_images)
all_labels = np.array(empty_labels + occupied_labels)

print(f"\nTotal images : {len(all_images)}")
print(f"Image shape  : {all_images.shape}")


X_train, X_test, y_train, y_test = train_test_split(
    all_images, all_labels,
    test_size    = 0.2,    
    random_state = 42,
    shuffle      = True
)

print(f"\nTraining images   : {len(X_train)}")
print(f"Testing images    : {len(X_test)}")


os.makedirs("data", exist_ok=True)

np.save("data/X_train.npy", X_train)
np.save("data/X_test.npy",  X_test)
np.save("data/y_train.npy", y_train)
np.save("data/y_test.npy",  y_test)

print("\nData saved successfully!")
print("Files created:")
print("  → data/X_train.npy")
print("  → data/X_test.npy")
print("  → data/y_train.npy")
print("  → data/y_test.npy")
print("\nPreprocessing complete!")