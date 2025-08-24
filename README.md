# Project Face Analysis

A deep learning pipeline for real-time face analysis: **age group classification**, **gender prediction**, and **emotion recognition** using PyTorch, MediaPipe, and OpenCV.

## Features

- **Age Group Classification**: Predicts age into 8 bins (0-6, 7-12, ..., 60-114) using ResNet-18.
- **Gender Prediction**: Binary gender classification (male/female) using ResNet-18.
- **Emotion Recognition**: Classifies facial emotion (angry, happy, neutral, sad, surprise) using ResNet-18.
- **Real-Time Detection**: Live webcam dashboard with MediaPipe face detection and PyTorch models.
- **Dataset Preparation**: Scripts for cleaning, resizing, and balancing UTKFace and FER-2013 datasets.
- **Jupyter Notebook**: Test models on custom images with visualization.

## Directory Structure

```
project_face_analysis/
│
├── datasets/                # Raw and cleaned datasets (not included in repo)
│   ├── raw/
│   └── clean/
│
├── models/                  # Saved PyTorch model weights
│
├── notebook/
│   ├── testing.ipynb        # Notebook for testing models on images
│
├── program/
│   ├── prepare_dataset.py   # Dataset cleaning and balancing
│   ├── train_age.py         # Age group model training
│   ├── train_gender.py      # Gender model training
│   ├── train_emotions.py    # Emotion model training
│
├── real_time_detection/
│   └── real_time_detection_program.py  # Tkinter dashboard for live detection
│
├── test/                    # Place test images here for notebook testing
│
├── requirements.txt         # Python dependencies
├── config.py                # Utility for checking GPU/torch setup
└── README.md
```

## Setup

1. **Clone the repository**  
   ```
   git clone https://github.com/KenROCKLEE/project_face_analysis.git
   cd project_face_analysis

2. **Create & activate a virtual environment**
    ```
    python -m venv .venv
    .venv\Scripts\activate
    (Python version should be 3.11)
    ```
3. **Install dependencies**  
   ```
   pip install -r requirements.txt
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129 (if using CUDA)
   ```

4. **Prepare datasets**  
   - Download datasets:
   - [UTKFace]
   - (https://www.kaggle.com/datasets/jangedoo/utkface-new)
   - [Facial Emotion Recognition Image Dataset]
   - (https://www.kaggle.com/datasets/sujaykapadnis/emotion-recognition-dataset)
   - Place raw images in `datasets/raw/UTKFace`.
   - Place clean images in `datasets/clean/emotion`. (Use only Angry,Happy,Neutral,Sad,Surprise)
   - Run dataset preparation:
     ```
     python program/prepare_dataset.py
     ```
   - This will create balanced, resized images in `datasets/clean/`.
   - 
5. **Train models**  
   - Age:
     ```
     python program/train_age.py
     ```
   - Gender:
     ```
     python program/train_gender.py
     ```
   - Emotion:
     ```
     python program/train_emotions.py
     ```
   - Trained models are saved in the `models/` directory. (If missing, create it manually.)

## Usage

### 1. Real-Time Face Analysis Dashboard

Run the Tkinter dashboard for live webcam detection:

```
python real_time_detection/real_time_detection_program.py
```

- Requires a webcam.
- Displays age group, gender, and emotion predictions for detected faces.
- Manually drag all the model to real_time_detection folder.

### 2. Test on Custom Images (Jupyter Notebook)

- Place test images in the `test/` directory.
- Open and run `notebook/testing.ipynb` to visualize predictions.

## Model Details

- **Architecture**: All models use ResNet-18 with custom classifier heads.
- **Input Size**: 128x128 RGB images.
- **Training**
  - Age & Emotion models → trained in two phases (feature extraction, then fine-tuning).
  - Gender model → trained in single phase (end-to-end fine-tuning).
- **Face Detection**: MediaPipe for robust, fast face localization.

## Credits

- [UTKFace Dataset](https://www.kaggle.com/datasets/jangedoo/utkface-new)
- [Facial Emotion Recognition Image Dataset](https://www.kaggle.com/datasets/sujaykapadnis/emotion-recognition-dataset)
- [PyTorch](https://pytorch.org/)
- [MediaPipe](https://mediapipe.dev/)
- [OpenCV](https://opencv.org/)

---

**Note:**  
- Datasets are not included due to size and license.  
- For best results, ensure your webcam is well-lit and faces are clearly visible.