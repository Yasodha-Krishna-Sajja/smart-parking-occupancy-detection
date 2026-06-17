import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import os

print("Loading preprocessed data...")

X_train = np.load("data/X_train.npy")
X_test  = np.load("data/X_test.npy")
y_train = np.load("data/y_train.npy")
y_test  = np.load("data/y_test.npy")

print(f"Training images : {X_train.shape}")
print(f"Testing images  : {X_test.shape}")
print(f"Training labels : {y_train.shape}")
print(f"Testing labels  : {y_test.shape}")


print("\nBuilding CNN model...")

model = Sequential([

    
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),


    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),

    
    Dense(1, activation='sigmoid')
])


model.compile(
    optimizer = 'adam',
    loss      = 'binary_crossentropy',
    metrics   = ['accuracy']
)


model.summary()

os.makedirs("models", exist_ok=True)


checkpoint = ModelCheckpoint(
    "models/parking_model.h5",
    monitor   = 'val_accuracy',
    save_best_only = True,
    verbose   = 1
)


early_stopping = EarlyStopping(
    monitor  = 'val_accuracy',
    patience = 5,
    verbose  = 1
)


print("\nStarting training...")

history = model.fit(
    X_train, y_train,
    epochs          = 20,
    batch_size      = 32,
    validation_data = (X_test, y_test),
    callbacks       = [checkpoint, early_stopping],
    verbose         = 1
)


print("\nEvaluating model...")

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy * 100:.2f}%")


os.makedirs("outputs/logs", exist_ok=True)


plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],     label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("outputs/logs/training_graphs.png")
plt.show()

print("\nTraining graphs saved to outputs/logs/training_graphs.png")
print("\nStep 4 Complete!")

