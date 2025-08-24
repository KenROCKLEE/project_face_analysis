import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import threading
import time
import os
import mediapipe as mp

class ResNetModel(nn.Module):
    def __init__(self, output_units=1, task='age'):
        super().__init__()
        if task == 'age':
            # Age classification model - matching your training code
            self.resnet = models.resnet18(weights=None)
            num_features = self.resnet.fc.in_features
            self.resnet.fc = nn.Sequential(
                nn.Dropout(0.6),
                nn.Linear(num_features, output_units)  # output_units = 8 for age classes
            )
        elif task == 'gender':
            # Gender model - matching your new training code
            self.resnet = models.resnet18(weights=None)
            num_features = self.resnet.fc.in_features
            self.resnet.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, output_units)  # output_units = 1 for binary classification
            )
        else:
            # Emotion models - keep original architecture
            self.resnet = models.resnet18(weights=None)
            num_features = self.resnet.fc.in_features
            self.resnet.fc = nn.Linear(num_features, output_units)
        self.task = task
        
    def forward(self, x):
        return self.resnet(x)

class FaceInfoDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Info Dashboard")
        self.root.geometry("1100x700")
        self.root.configure(bg='#2b2b2b')
        
        # Model configuration
        self.IMG_SIZE = (128, 128)
        
        # Age groups - matching your training code
        self.AGE_GROUPS = [
            (0, 6), (7, 12), (13, 19),
            (20, 29), (30, 39), (40, 49), (50, 59), (60, 114)
        ]
        self.AGE_CLASS_LABELS = [f"{start}-{end}" for start, end in self.AGE_GROUPS]
        self.NUM_AGE_CLASSES = len(self.AGE_CLASS_LABELS)
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 for short-range (< 2 meters), 1 for full-range
            min_detection_confidence=0.7
        )
        
        # Initialize models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.models = {}
        self.load_models()
        
        # Initialize camera
        self.cap = None
        self.camera_running = False
        
        # Face analysis results
        self.current_results = {
            'age': 'N/A',
            'gender': 'N/A',
            'emotion': 'N/A'
        }
        
        # Face detection parameters
        self.face_detected = False
        self.detection_counter = 0
        self.face_confidence = 0.0
        
        # Define transforms - matching your training codes
        self.normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                       std=[0.229, 0.224, 0.225])
        
        # Age model uses normalized transforms (like your training)
        self.age_transform = transforms.Compose([
            transforms.Resize(self.IMG_SIZE),
            transforms.ToTensor(),
            self.normalize_transform
        ])
        
        # Gender model also uses normalized transforms (matching your new training)
        self.gender_transform = transforms.Compose([
            transforms.Resize(self.IMG_SIZE),
            transforms.ToTensor(),
            self.normalize_transform
        ])
        
        # Keep original transforms for emotion model
        self.test_transform_no_norm = transforms.Compose([
            transforms.Resize(self.IMG_SIZE),
            transforms.ToTensor()
        ])
        
        self.test_transform_norm = transforms.Compose([
            transforms.Resize(self.IMG_SIZE),
            transforms.ToTensor(),
            self.normalize_transform
        ])
        
        # Label mappings
        self.gender_map = {0: 'Female', 1: 'Male'}
        self.emotion_map = {0: 'Angry 😠', 1: 'Happy 😀', 2: 'Neutral 😐', 3: 'Sad 😢', 4: 'Surprised 😲'}
        
        self.setup_ui()
        
    def load_models(self):
        """Load the three trained models"""
        try:
            # Get script directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Model paths
            model_paths = {
                'age': os.path.join(script_dir, 'age_model_resnet.pth'),
                'gender': os.path.join(script_dir, 'gender_model_resnet.pth'),
                'emotion': os.path.join(script_dir, 'emotion_model_resnet.pth')
            }
            
            loaded_models = []
            
            for model_name, path in model_paths.items():
                if os.path.exists(path):
                    try:
                        # Create models with appropriate architectures
                        if model_name == 'age':
                            self.models[model_name] = ResNetModel(output_units=self.NUM_AGE_CLASSES, task='age')
                        elif model_name == 'gender':
                            self.models[model_name] = ResNetModel(output_units=1, task='gender')
                        else:  # emotion
                            self.models[model_name] = ResNetModel(output_units=5, task='emotion')
                        
                        # Load weights and move to device
                        self.models[model_name].load_state_dict(torch.load(path, map_location=self.device))
                        self.models[model_name].to(self.device)
                        self.models[model_name].eval()
                        loaded_models.append(model_name)
                        print(f"✓ {model_name.capitalize()} model loaded successfully")
                    except Exception as e:
                        print(f"✗ Failed to load {model_name} model: {e}")
                else:
                    print(f"✗ Model file not found: {path}")
            
            if loaded_models:
                print(f"Successfully loaded {len(loaded_models)} model(s): {', '.join(loaded_models)}")
            else:
                print("No models loaded. UI will work but predictions will show 'No Model'")
                
        except Exception as e:
            print(f"Model loading error: {e}")
            self.models = {}
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main container
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(main_frame, text="Face Info Dashboard - MediaPipe Enhanced", 
                              font=('Arial', 22, 'bold'), fg='white', bg='#2b2b2b')
        title_label.pack(pady=(0, 20))
        
        # Status label
        self.status_label = tk.Label(main_frame, text="Ready - Click Start Camera to begin", 
                                    font=('Arial', 10), fg='#cccccc', bg='#2b2b2b')
        self.status_label.pack(pady=(0, 10))
        
        # Content frame
        content_frame = tk.Frame(main_frame, bg='#2b2b2b')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left side - Video feed
        video_frame = tk.Frame(content_frame, bg='#3b3b3b', relief=tk.RAISED, bd=2)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        video_title = tk.Label(video_frame, text="Webcam Feed", 
                              font=('Arial', 16, 'bold'), fg='white', bg='#3b3b3b')
        video_title.pack(pady=10)
        
        self.video_label = tk.Label(video_frame, bg='black', text="Camera Off\n\nClick 'Start Camera' to begin",
                                   fg='white', font=('Arial', 14))
        self.video_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Camera controls
        control_frame = tk.Frame(video_frame, bg='#3b3b3b')
        control_frame.pack(pady=10)
        
        self.start_btn = tk.Button(control_frame, text="Start Camera", 
                                  command=self.start_camera, font=('Arial', 12),
                                  bg='#4CAF50', fg='white', relief=tk.FLAT, padx=20)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(control_frame, text="Stop Camera", 
                                 command=self.stop_camera, font=('Arial', 12),
                                 bg='#f44336', fg='white', relief=tk.FLAT, padx=20,
                                 state='disabled')
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Right side - Face info panel
        info_frame = tk.Frame(content_frame, bg='#3b3b3b', relief=tk.RAISED, bd=2)
        info_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        info_frame.configure(width=280)
        
        info_title = tk.Label(info_frame, text="Face Analysis", 
                             font=('Arial', 16, 'bold'), fg='white', bg='#3b3b3b')
        info_title.pack(pady=20)
        
        # Instruction
        instruction = tk.Label(info_frame, text="Don't take it seriously!\nModels can be inaccurate.", 
                              font=('Arial', 10, 'italic'), fg='#cccccc', bg='#3b3b3b',
                              justify=tk.CENTER)
        instruction.pack(pady=(0, 20))
        
        # Face detection status
        self.face_status = tk.Label(info_frame, text="👤 No face detected", 
                                   font=('Arial', 11), fg='#ff6b6b', bg='#3b3b3b')
        self.face_status.pack(pady=(0, 10))
        
        # Detection confidence
        self.confidence_status = tk.Label(info_frame, text="Detection: 0%", 
                                         font=('Arial', 9), fg='#888888', bg='#3b3b3b')
        self.confidence_status.pack(pady=(0, 20))
        
        # Info display
        self.create_info_display(info_frame)
        
        # Model status
        model_status_frame = tk.Frame(info_frame, bg='#3b3b3b')
        model_status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        tk.Label(model_status_frame, text="Models Loaded:", 
                font=('Arial', 9, 'bold'), fg='#cccccc', bg='#3b3b3b').pack()
        
        status_text = []
        for model_name in ['age', 'gender', 'emotion']:
            if model_name in self.models:
                status_text.append(f"✓ {model_name.capitalize()}")
            else:
                status_text.append(f"✗ {model_name.capitalize()}")
        
        tk.Label(model_status_frame, text='\n'.join(status_text), 
                font=('Arial', 8), fg='#cccccc', bg='#3b3b3b').pack()
        
        # MediaPipe info
        mp_info = tk.Label(model_status_frame, text="🚀 MediaPipe Face Detection", 
                          font=('Arial', 8, 'italic'), fg='#4CAF50', bg='#3b3b3b')
        mp_info.pack(pady=(10, 0))
        
    def create_info_display(self, parent):
        """Create the information display panel"""
        # Age
        age_frame = tk.Frame(parent, bg='#4b4b4b', relief=tk.RAISED, bd=1)
        age_frame.pack(fill=tk.X, padx=20, pady=5)
        
        tk.Label(age_frame, text="Age:", font=('Arial', 12, 'bold'), 
                fg='white', bg='#4b4b4b').pack(side=tk.LEFT, padx=10, pady=10)
        
        self.age_label = tk.Label(age_frame, text="N/A", font=('Arial', 11), 
                                 fg='#4CAF50', bg='#4b4b4b')
        self.age_label.pack(side=tk.RIGHT, padx=10, pady=10)
        
        # Gender
        gender_frame = tk.Frame(parent, bg='#4b4b4b', relief=tk.RAISED, bd=1)
        gender_frame.pack(fill=tk.X, padx=20, pady=5)
        
        tk.Label(gender_frame, text="Gender:", font=('Arial', 12, 'bold'), 
                fg='white', bg='#4b4b4b').pack(side=tk.LEFT, padx=10, pady=10)
        
        self.gender_label = tk.Label(gender_frame, text="N/A", font=('Arial', 11), 
                                    fg='#2196F3', bg='#4b4b4b')
        self.gender_label.pack(side=tk.RIGHT, padx=10, pady=10)
        
        # Emotion
        emotion_frame = tk.Frame(parent, bg='#4b4b4b', relief=tk.RAISED, bd=1)
        emotion_frame.pack(fill=tk.X, padx=20, pady=5)
        
        tk.Label(emotion_frame, text="Emotion:", font=('Arial', 12, 'bold'), 
                fg='white', bg='#4b4b4b').pack(side=tk.LEFT, padx=10, pady=10)
        
        self.emotion_label = tk.Label(emotion_frame, text="N/A", font=('Arial', 11), 
                                     fg='#FF9800', bg='#4b4b4b')
        self.emotion_label.pack(side=tk.RIGHT, padx=10, pady=10)
    
    def start_camera(self):
        """Start the camera feed"""
        if not self.camera_running:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                # Set camera properties for better performance
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                
                self.camera_running = True
                self.start_btn.configure(state='disabled')
                self.stop_btn.configure(state='normal')
                self.status_label.configure(text="Camera active - Looking for faces with MediaPipe...")
                self.update_frame()
            else:
                messagebox.showerror("Error", "Could not open camera")
    
    def stop_camera(self):
        """Stop the camera feed"""
        self.camera_running = False
        if self.cap:
            self.cap.release()
        self.start_btn.configure(state='normal')
        self.stop_btn.configure(state='disabled')
        self.status_label.configure(text="Camera stopped")
        
        # Clear video display
        self.video_label.configure(image='', text="Camera Off\n\nClick 'Start Camera' to begin")
        self.video_label.image = None
        
        # Reset results
        self.current_results = {'age': 'N/A', 'gender': 'N/A', 'emotion': 'N/A'}
        self.face_detected = False
        self.face_confidence = 0.0
        self.update_display()
    
    def update_frame(self):
        """Update video frame and perform face analysis"""
        if self.camera_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Perform face detection and analysis
                self.analyze_frame(frame.copy())
                
                # Convert frame for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                # Resize frame to fit display
                display_width = 480
                display_height = 360
                frame_pil = frame_pil.resize((display_width, display_height), Image.Resampling.LANCZOS)
                
                frame_tk = ImageTk.PhotoImage(frame_pil)
                self.video_label.configure(image=frame_tk, text="")
                self.video_label.image = frame_tk
                
                # Schedule next frame update
                self.root.after(30, self.update_frame)
    
    def analyze_frame(self, frame):
        """Analyze the frame for face info using MediaPipe"""
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.face_detection.process(rgb_frame)
        
        if results.detections:
            self.face_detected = True
            
            # Get the most confident detection
            best_detection = max(results.detections, key=lambda d: d.score[0])
            self.face_confidence = best_detection.score[0]
            
            # Update face status
            self.face_status.configure(text="👤 Face detected", fg='#4CAF50')
            self.confidence_status.configure(text=f"Detection: {self.face_confidence*100:.1f}%", fg='#4CAF50')
            
            # Extract face bounding box
            h, w, _ = frame.shape
            bbox = best_detection.location_data.relative_bounding_box
            
            # Convert relative coordinates to absolute
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # Add padding and ensure bounds
            padding_x = int(0.2 * width)
            padding_y = int(0.2 * height)
            x1 = max(0, x - padding_x)
            y1 = max(0, y - padding_y)
            x2 = min(w, x + width + padding_x)
            y2 = min(h, y + height + padding_y)
            
            # Extract face region
            face_roi = rgb_frame[y1:y2, x1:x2]
            
            if face_roi.size > 0 and face_roi.shape[0] > 30 and face_roi.shape[1] > 30:
                # Only run prediction every few frames for performance
                self.detection_counter += 1
                if self.detection_counter % 8 == 0:  # Every 8th frame (slightly faster due to better detection)
                    self.predict_face_attributes(face_roi)
        else:
            self.face_detected = False
            self.face_confidence = 0.0
            self.face_status.configure(text="👤 No face detected", fg='#ff6b6b')
            self.confidence_status.configure(text="Detection: 0%", fg='#888888')
            
            # Don't immediately clear results - keep last known values for a bit
            if not hasattr(self, 'no_face_counter'):
                self.no_face_counter = 0
            self.no_face_counter += 1
            
            # Clear results after not seeing a face for a while
            if self.no_face_counter > 90:  # ~3 seconds at 30fps
                self.current_results = {'age': 'N/A', 'gender': 'N/A', 'emotion': 'N/A'}
                self.no_face_counter = 0
        
        self.update_display()
    
    def predict_face_attributes(self, face_roi):
        """Predict age, gender, and emotion from face ROI"""
        # Check if models are loaded
        if not self.models:
            self.current_results = {'age': 'No Model', 'gender': 'No Model', 'emotion': 'No Model'}
            return
            
        try:
            # Convert numpy array to PIL Image (face_roi is already RGB)
            if face_roi.shape[0] < 20 or face_roi.shape[1] < 20:
                return  # Face too small
                
            pil_face = Image.fromarray(face_roi)
            
            # Adjust confidence thresholds based on face detection confidence
            base_threshold = 0.3
            detection_bonus = self.face_confidence * 0.2  # Boost confidence if face detection is strong
            
            with torch.no_grad():
                # Age prediction (classification) - using normalized input
                if 'age' in self.models:
                    img_tensor_age = self.age_transform(pil_face).unsqueeze(0).to(self.device)
                    age_logits = self.models['age'](img_tensor_age)
                    age_probs = torch.softmax(age_logits, dim=1)
                    
                    # Get top-2 predictions
                    top2_probs, top2_indices = torch.topk(age_probs, 2, dim=1)
                    top1_idx = top2_indices[0][0].item()
                    top1_prob = top2_probs[0][0].item()
                    
                    # Use adaptive threshold
                    age_threshold = max(0.25, base_threshold - detection_bonus)
                    
                    if top1_prob > age_threshold:
                        age_range = self.AGE_CLASS_LABELS[top1_idx]
                        confidence = top1_prob * 100
                        self.current_results['age'] = f"{age_range} ({confidence:.0f}%)"
                    else:
                        # Show top-2 if not confident
                        top2_idx = top2_indices[0][1].item()
                        age_range1 = self.AGE_CLASS_LABELS[top1_idx]
                        age_range2 = self.AGE_CLASS_LABELS[top2_idx]
                        self.current_results['age'] = f"{age_range1} or {age_range2}"
                else:
                    self.current_results['age'] = 'No Model'
                
                # Gender prediction - updated to match your new training approach
                if 'gender' in self.models:
                    img_tensor_gender = self.gender_transform(pil_face).unsqueeze(0).to(self.device)
                    gender_logits = self.models['gender'](img_tensor_gender).squeeze()
                    gender_prob = torch.sigmoid(gender_logits).item()
                    gender_idx = int(round(gender_prob))
                    
                    # Show confidence with adaptive threshold
                    confidence = max(gender_prob, 1 - gender_prob) * 100
                    gender_threshold = max(55, 65 - detection_bonus * 100)
                    
                    if confidence > gender_threshold:
                        gender_label = self.gender_map.get(gender_idx, 'Unknown')
                        self.current_results['gender'] = f"{gender_label} ({confidence:.0f}%)"
                    else:
                        gender_label = self.gender_map.get(gender_idx, 'Unknown')
                        self.current_results['gender'] = f"{gender_label}"
                else:
                    self.current_results['gender'] = 'No Model'
                
                # Emotion prediction (keep original approach but with adaptive threshold)
                if 'emotion' in self.models:
                    img_tensor_norm = self.test_transform_norm(pil_face).unsqueeze(0).to(self.device)
                    emotion_logits = self.models['emotion'](img_tensor_norm)
                    emotion_probs = torch.softmax(emotion_logits, dim=1)
                    emotion_idx = int(torch.argmax(emotion_probs, dim=1).item())
                    emotion_label = self.emotion_map.get(emotion_idx, 'Unknown')
                    
                    # Adaptive threshold for emotion
                    max_prob = emotion_probs[0][emotion_idx].item()
                    emotion_threshold = max(0.35, 0.45 - detection_bonus)
                    
                    if max_prob > emotion_threshold:
                        confidence = max_prob * 100
                        self.current_results['emotion'] = f"{emotion_label} ({confidence:.0f}%)"
                    else:
                        self.current_results['emotion'] = 'Uncertain'
                else:
                    self.current_results['emotion'] = 'No Model'
                
        except Exception as e:
            print(f"Prediction error: {e}")
            self.current_results = {'age': 'Error', 'gender': 'Error', 'emotion': 'Error'}
    
    def update_display(self):
        """Update the information display"""
        self.age_label.configure(text=self.current_results['age'])
        self.gender_label.configure(text=self.current_results['gender'])
        self.emotion_label.configure(text=self.current_results['emotion'])
    
    def __del__(self):
        """Cleanup when application closes"""
        if self.cap:
            self.cap.release()
        if hasattr(self, 'face_detection'):
            self.face_detection.close()

def main():
    root = tk.Tk()
    app = FaceInfoDashboard(root)
    
    # Handle window close
    def on_closing():
        app.stop_camera()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()