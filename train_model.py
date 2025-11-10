import os
import numpy as np
import librosa
import librosa.display
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.image import resize

# -----------------------------
# Parameters
# -----------------------------
DATA_DIR = 'audio_dataset/'  # folder structure: audio_dataset/C/key1.mp3 etc.
SR = 22050
DURATION = 30  # seconds
N_MELS = 128
IMG_WIDTH = 256  # resized width (time steps)
IMG_HEIGHT = N_MELS

# -----------------------------
# Load audio files and labels
# -----------------------------
file_paths = []
labels = []

for key in os.listdir(DATA_DIR):
    key_path = os.path.join(DATA_DIR, key)
    if os.path.isdir(key_path):
        for file in os.listdir(key_path):
            if file.endswith('.mp3') or file.endswith('.wav'):
                file_paths.append(os.path.join(key_path, file))
                labels.append(key)

print(f"Found {len(file_paths)} audio files.")

# -----------------------------
# Process audio into spectrograms
# -----------------------------
def audio_to_melspectrogram(file_path):
    y, sr = librosa.load(file_path, sr=SR)
    desired_length = SR * DURATION
    if len(y) < desired_length:
        y = np.pad(y, (0, desired_length - len(y)))
    else:
        y = y[:desired_length]
    
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())
    
    # Resize to fixed width
    S_resized = resize(S_norm[..., np.newaxis], [IMG_HEIGHT, IMG_WIDTH])
    S_img = np.repeat(S_resized, 3, axis=-1)  # 3 channels
    return S_img.numpy()

# Convert all files
X = np.array([audio_to_melspectrogram(f) for f in file_paths])

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(labels)
y_categorical = to_categorical(y_encoded, num_classes=len(le.classes_))

# -----------------------------
# Train/val/test split
# -----------------------------
X_train, X_temp, y_train, y_temp = train_test_split(X, y_categorical, test_size=0.3, random_state=42, stratify=y_categorical)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# -----------------------------
# Build CNN model
# -----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# -----------------------------
# Train model
# -----------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=50,
    callbacks=[early_stop]
)

# -----------------------------
# Evaluate and save
# -----------------------------
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.2f}")

model.save("music_key_classifier.h5")
print("Model saved as music_key_classifier.h5")