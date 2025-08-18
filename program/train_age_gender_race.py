"""
UTKFace Age, Gender, Race Prediction Pipeline
Author: Your Name
Date: 2025-08-18

This script loads the UTKFace dataset, preprocesses images, builds and trains three separate models for age (regression), gender (binary classification), and race (multiclass classification), evaluates them, and saves the results and models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Constants
IMG_SIZE = (128, 128)
DATASET_DIR = 'datasets/raw/UTKFace'
MODEL_DIR = 'models'

# 1. Load dataset and parse labels
def load_dataset(dataset_dir=DATASET_DIR, img_size=IMG_SIZE, max_images=None):
    X, y_age, y_gender, y_race = [], [], [], []
    files = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg')]
    if max_images:
        files = files[:max_images]
    for fname in files:
        try:
            age, gender, race, _ = fname.split('_', 3)
            img_path = os.path.join(dataset_dir, fname)
            img = load_img(img_path, target_size=img_size)
            img = img_to_array(img)
            X.append(img)
            y_age.append(int(age))
            y_gender.append(int(gender))
            y_race.append(int(race))
        except Exception as e:
            print(f"Skipping {fname}: {e}")
    X = np.array(X, dtype='float32')
    y_age = np.array(y_age)
    y_gender = np.array(y_gender)
    y_race = np.array(y_race)
    return X, y_age, y_gender, y_race

# 2. Preprocess images
def preprocess_images(X):
    X = X / 255.0  # Normalize to [0,1]
    return X

# 3. Split dataset
def split_dataset(X, y_age, y_gender, y_race, test_size=0.2, val_size=0.1, random_state=42):
    X_train, X_test, y_age_train, y_age_test, y_gender_train, y_gender_test, y_race_train, y_race_test = train_test_split(
        X, y_age, y_gender, y_race, test_size=test_size, random_state=random_state)
    X_train, X_val, y_age_train, y_age_val, y_gender_train, y_gender_val, y_race_train, y_race_val = train_test_split(
        X_train, y_age_train, y_gender_train, y_race_train, test_size=val_size, random_state=random_state)
    return (X_train, X_val, X_test, y_age_train, y_age_val, y_age_test,
            y_gender_train, y_gender_val, y_gender_test, y_race_train, y_race_val, y_race_test)

# 4. Model builders
def build_base_cnn(output_units, output_activation):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        MaxPooling2D(2,2),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        BatchNormalization(),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        BatchNormalization(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(output_units, activation=output_activation)
    ])
    return model

def build_age_model():
    return build_base_cnn(1, 'linear')

def build_gender_model():
    return build_base_cnn(1, 'sigmoid')

def build_race_model():
    return build_base_cnn(5, 'softmax')

# 5. Training and evaluation
def train_and_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test, model_path, task='age', epochs=10, batch_size=32):
    if task == 'age':
        loss = 'mse'
        metrics = ['mae']
    elif task == 'gender':
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    elif task == 'race':
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
    else:
        raise ValueError('Unknown task')

    model.compile(optimizer=Adam(), loss=loss, metrics=metrics)
    checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min')
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint],
        verbose=2
    )
    model.load_weights(model_path)
    results = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test results for {task}: {dict(zip(model.metrics_names, results))}")
    return history, results

# 6. Plot training history
def plot_history(history, task, save_dir='models'):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title(f'{task} Loss')
    plt.legend()
    plt.subplot(1,2,2)
    metric = 'accuracy' if 'accuracy' in history.history else 'mae'
    plt.plot(history.history[metric], label='train')
    plt.plot(history.history['val_' + metric], label='val')
    plt.title(f'{task} {metric.title()}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{task}_history.png'))
    plt.close()

# 7. Main pipeline
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print('Loading dataset...')
    X, y_age, y_gender, y_race = load_dataset()
    print(f'Total samples: {len(X)}')
    X = preprocess_images(X)
    y_gender_bin = y_gender
    y_race_cat = to_categorical(y_race, num_classes=5)
    (X_train, X_val, X_test, y_age_train, y_age_val, y_age_test,
     y_gender_train, y_gender_val, y_gender_test, y_race_train, y_race_val, y_race_test) = split_dataset(
        X, y_age, y_gender_bin, y_race_cat)

    # Age model
    print('Training age model...')
    age_model = build_age_model()
    age_history, age_results = train_and_evaluate(
        age_model, X_train, y_age_train, X_val, y_age_val, X_test, y_age_test,
        os.path.join(MODEL_DIR, 'age_model.h5'), task='age')
    plot_history(age_history, 'age', save_dir=MODEL_DIR)

    # Gender model
    print('Training gender model...')
    gender_model = build_gender_model()
    gender_history, gender_results = train_and_evaluate(
        gender_model, X_train, y_gender_train, X_val, y_gender_val, X_test, y_gender_test,
        os.path.join(MODEL_DIR, 'gender_model.h5'), task='gender')
    plot_history(gender_history, 'gender', save_dir=MODEL_DIR)

    # Race model
    print('Training race model...')
    race_model = build_race_model()
    race_history, race_results = train_and_evaluate(
        race_model, X_train, y_race_train, X_val, y_race_val, X_test, y_race_test,
        os.path.join(MODEL_DIR, 'race_model.h5'), task='race')
    plot_history(race_history, 'race', save_dir=MODEL_DIR)

    print('All models trained and saved.')

if __name__ == '__main__':
    main()
