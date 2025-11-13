import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.image import resize
import tensorflow as tf

# -----------------------------
# Parameters
# -----------------------------
DATA_DIR = "genres_original"
SR = 22050
DURATION = 30
N_MELS = 128
IMG_WIDTH = 256
IMG_HEIGHT = N_MELS

GENRES = ['blues', 'classical', 'hiphop', 'disco', 'country', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# -----------------------------
# Audio to Mel Spectrogram
# -----------------------------
def audio_to_melspectrogram(file_path, augment=False):
    try:
        y, sr = librosa.load(file_path, sr=SR, mono=True)
    except Exception as e:
        print(f"Skipping {file_path}: {e}")
        return None

    # optional augmentation
    if augment:
        if np.random.rand() < 0.5:
            y = np.roll(y, np.random.randint(len(y)//10))  # time shift
        if np.random.rand() < 0.3:
            y = librosa.effects.pitch_shift(y, sr, n_steps=np.random.uniform(-1, 1))

    desired_length = SR * DURATION
    if len(y) < desired_length:
        y = np.pad(y, (0, desired_length - len(y)))
    else:
        y = y[:desired_length]

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())
    S_resized = resize(S_norm[..., np.newaxis], [IMG_HEIGHT, IMG_WIDTH]).numpy()
    S_img = np.repeat(S_resized, 3, axis=-1)
    return S_img.astype(np.float32)

# -----------------------------
# Load dataset
# -----------------------------
file_paths, labels = [], []
for genre in GENRES:
    genre_path = os.path.join(DATA_DIR, genre)
    for file in os.listdir(genre_path):
        if file.lower().endswith(".wav"):
            file_paths.append(os.path.join(genre_path, file))
            labels.append(genre)

X_list, labels_clean = [], []
for f, label in zip(file_paths, labels):
    S = audio_to_melspectrogram(f)
    if S is not None and S.shape == (IMG_HEIGHT, IMG_WIDTH, 3):
        X_list.append(S)
        labels_clean.append(label)

X = np.stack(X_list)
labels = labels_clean
print("Spectrograms ready:", X.shape)

# global normalization
X = (X - np.mean(X)) / np.std(X)

# -----------------------------
# Encode labels
# -----------------------------
le = LabelEncoder()
le.fit(GENRES)
y_encoded = le.transform(labels)
y_categorical = to_categorical(y_encoded, num_classes=len(GENRES))

# -----------------------------
# Split
# -----------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_categorical, test_size=0.3, random_state=42, stratify=y_categorical
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Data augmentation on training set
def augment_batch(X_batch, y_batch):
    X_aug, y_aug = [], []
    for x, y in zip(X_batch, y_batch):
        if np.random.rand() < 0.4:  # augment 40% of samples
            x_aug = audio_to_melspectrogram(file_paths[np.random.randint(len(file_paths))], augment=True)
            if x_aug is not None:
                X_aug.append(x_aug)
                y_aug.append(y)
        else:
            X_aug.append(x)
            y_aug.append(y)
    return np.array(X_aug), np.array(y_aug)

# -----------------------------
# CNN Model
# -----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
           kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(GENRES), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

# -----------------------------
# Train
# -----------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=100,
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------
# Evaluate
# -----------------------------
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"✅ Test accuracy: {test_acc:.3f}")

model.save("music_genre_classifier_fixed.keras")
print("Model saved.")
