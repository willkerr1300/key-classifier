import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.image import resize
import tensorflow as tf

# -----------------------------
# Parameters
# -----------------------------
DATA_DIR = "audio_dataset"     # audio_dataset/genre_name/*.wav
SR = 22050                     # sample rate
DURATION = 30                  # seconds — if shorter, padded
N_MELS = 128                   # mel bands
IMG_WIDTH = 256               # spectrogram time width (resize)
IMG_HEIGHT = N_MELS            # spectrogram height

# Explicit genre class list (GTZAN standard)
GENRES = ['blues', 'classical', 'hiphop', 'disco', 'country', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# -----------------------------
# Collect file paths and labels
# -----------------------------
file_paths = []
labels = []

for genre in GENRES:
    genre_path = os.path.join(DATA_DIR, genre)
    if os.path.isdir(genre_path):
        for file in os.listdir(genre_path):
            if file.lower().endswith(('.mp3', '.wav', '.ogg')):
                file_paths.append(os.path.join(genre_path, file))
                labels.append(genre)

print(f"Found {len(file_paths)} audio files.")

# -----------------------------
# Convert audio → mel spectrogram image
# -----------------------------
def audio_to_melspectrogram(file_path):
    y, sr = librosa.load(file_path, sr=SR)
    
    # Ensure fixed duration
    desired_length = SR * DURATION
    if len(y) < desired_length:
        y = np.pad(y, (0, desired_length - len(y)))
    else:
        y = y[:desired_length]

    # Mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Normalize 0–1
    S_norm = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())

    # Resize to fixed shape
    S_resized = resize(S_norm[..., np.newaxis], [IMG_HEIGHT, IMG_WIDTH])
    
    # Convert to 3-channel image
    S_img = np.repeat(S_resized, 3, axis=-1)

    return S_img.numpy()

print("Processing audio to spectrograms...")
X = np.array([audio_to_melspectrogram(f) for f in file_paths])
print("Spectrograms ready:", X.shape)

# -----------------------------
# Encode labels
# -----------------------------
le = LabelEncoder()
le.fit(GENRES)
y_encoded = le.transform(labels)
y_categorical = to_categorical(y_encoded, num_classes=len(GENRES))

# -----------------------------
# Train / Validation / Test Split
# -----------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_categorical, test_size=0.3, random_state=42, stratify=y_categorical
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"📊 Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# -----------------------------
# CNN Model
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
    Dense(len(GENRES), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -----------------------------
# Train
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
# Evaluate & Save
# -----------------------------
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.3f}")

model.save("music_genre_classifier.h5")
print("Model saved: music_genre_classifier.h5")
