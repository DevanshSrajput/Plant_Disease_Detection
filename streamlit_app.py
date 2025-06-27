# Suppress all warnings at the very beginning
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Standard imports
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import pandas as pd
import json
import time
import sys
import threading
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Page configuration (must be first Streamlit command)
st.set_page_config(
    page_title="🌿 Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to import local modules with error handling
try:
    from src.data_preprocessing import DataPreprocessor
    from src.model import PlantDiseaseModel
    from src.train import ModelTrainer
    from src.evaluate import ModelEvaluator
    from config import PLANTVILLAGE_DIR, MODELS_DIR, validate_data_directory
    LOCAL_MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"⚠️ Some modules couldn't be imported: {e}")
    st.info("The app will run with basic functionality")
    LOCAL_MODULES_AVAILABLE = False
    # Set fallback values
    PLANTVILLAGE_DIR = Path("data/plantvillage")
    MODELS_DIR = Path("models")
    MODELS_DIR.mkdir(exist_ok=True)

# Custom CSS for modern UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main { padding-top: 1rem; }
    
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #2E7D32, #4CAF50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: fadeIn 1s ease-in;
    }
    
    .sub-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.8rem;
        font-weight: 600;
        color: #1B5E20;
        margin: 1.5rem 0;
        border-bottom: 3px solid #E8F5E8;
        padding-bottom: 0.5rem;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #f8fffe 0%, #e8f5e8 100%);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid #E0E0E0;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }
    
    .healthy-card {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        border-left: 6px solid #4CAF50;
    }
    
    .diseased-card {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left: 6px solid #f44336;
    }
    
    .training-card {
        background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
        border-left: 6px solid #ff9800;
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .progress-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 6px solid #2196F3;
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #fafafa 100%);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid #E0E0E0;
        margin: 0.5rem 0;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #4CAF50 0%, #66BB6A 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(76, 175, 80, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(76, 175, 80, 0.4);
    }
    
    .stSelectbox > div > div {
        border-radius: 12px;
        border: 2px solid #E0E0E0;
        font-family: 'Inter', sans-serif;
    }
    
    .stFileUploader > div {
        border-radius: 16px;
        border: 2px dashed #4CAF50;
        padding: 2rem;
        text-align: center;
        background: rgba(76, 175, 80, 0.05);
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: #2E7D32;
        background: rgba(76, 175, 80, 0.1);
    }
    
    .training-log {
        background: #1a1a1a;
        color: #00ff41;
        padding: 1.5rem;
        border-radius: 12px;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        max-height: 300px;
        overflow-y: auto;
        margin: 1rem 0;
        border: 1px solid #333;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .training-metrics {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .training-metric {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    
    .info-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border-left: 6px solid #2196F3;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(33, 150, 243, 0.1);
    }
    
    .warning-card {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border-left: 6px solid #ff9800;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(255, 152, 0, 0.1);
    }
    
    .error-card {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border-left: 6px solid #f44336;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(244, 67, 54, 0.1);
    }
    
    .success-card {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border-left: 6px solid #4CAF50;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(76, 175, 80, 0.1);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .training-active {
        animation: pulse 2s infinite;
    }
    
    /* Hide Streamlit elements */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    .stDeployButton { display: none; }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #4CAF50, #66BB6A);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #fafafa 0%, #f5f5f5 100%);
    }
</style>
""", unsafe_allow_html=True)

# Configuration
DATA_DIR = Path("data/plantvillage")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

class FastModelTrainer:
    """Ultra-fast model trainer optimized for quick results"""
    
    def __init__(self, data_path, models_dir):
        self.data_path = data_path
        self.models_dir = models_dir
        self.training_active = False
        
    def create_ultra_fast_model(self, num_classes, model_type="ultra_fast"):
        """Create ultra-lightweight models for very fast training"""
        if model_type == "ultra_fast":
            # Extremely lightweight CNN for demo purposes
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(16, (5, 5), activation='relu', input_shape=(128, 128, 3)),
                tf.keras.layers.MaxPooling2D((4, 4)),
                tf.keras.layers.Dropout(0.25),
                
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((4, 4)),
                tf.keras.layers.Dropout(0.25),
                
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(num_classes, activation='softmax')
            ])
        elif model_type == "efficient":
            # Optimized efficient CNN
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(num_classes, activation='softmax')
            ])
        else:
            # Fast transfer learning with MobileNetV2
            base_model = tf.keras.applications.MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(128, 128, 3),
                alpha=0.5
            )
            base_model.trainable = False
            
            model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(num_classes, activation='softmax')
            ])
        
        return model
    
    def prepare_data_ultra_fast(self):
        """Prepare data with better class balancing"""
        try:
            # Enhanced data generator with better preprocessing
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,  # Increased for better augmentation
                width_shift_range=0.15,
                height_shift_range=0.15,
                horizontal_flip=True,
                zoom_range=0.1,
                brightness_range=[0.9, 1.1],
                validation_split=0.2
            )
            
            # Training data
            train_generator = train_datagen.flow_from_directory(
                self.data_path,
                target_size=(128, 128),
                batch_size=16,  # Smaller batch size for better learning
                class_mode='categorical',
                subset='training',
                shuffle=True,
                seed=42
            )
            
            # Validation data
            val_generator = train_datagen.flow_from_directory(
                self.data_path,
                target_size=(128, 128),
                batch_size=16,
                class_mode='categorical',
                subset='validation',
                shuffle=False,
                seed=42
            )
            
            return train_generator, val_generator
            
        except Exception as e:
            st.error(f"Error preparing data: {e}")
            return None, None
    
    def train_model_with_progress(self, model_type="ultra_fast", epochs=5, progress_callback=None, log_callback=None):
        """Enhanced training with better convergence"""
        
        try:
            self.training_active = True
            
            if log_callback:
                log_callback("🚀 Starting enhanced training...")
                log_callback(f"📋 Configuration: {model_type} model, {epochs} epochs")
            
            # Prepare data
            train_gen, val_gen = self.prepare_data_ultra_fast()
            if train_gen is None or val_gen is None:
                raise ValueError("Failed to prepare data")
            
            num_classes = train_gen.num_classes
            class_names = list(train_gen.class_indices.keys())
            
            if log_callback:
                log_callback(f"✅ Data loaded: {num_classes} classes")
                log_callback(f"📂 Classes: {', '.join(class_names[:5])}...")
                log_callback(f"📈 Training samples: {train_gen.samples}")
                log_callback(f"📊 Validation samples: {val_gen.samples}")
            
            # Create model
            model = self.create_ultra_fast_model(num_classes, model_type)
            
            # Enhanced compilation with better optimizer settings
            if model_type == "ultra_fast":
                learning_rate = 0.001  # Reduced for better convergence
                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=learning_rate,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-7
                )
            else:
                learning_rate = 0.0005
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            
            model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            if log_callback:
                log_callback(f"📐 Model compiled: {model.count_params():,} parameters")
                log_callback(f"⚡ Learning rate: {learning_rate}")
            
            # Progress callback
            class ProgressCallback(tf.keras.callbacks.Callback):
                def __init__(self, progress_callback, log_callback, total_epochs):
                    self.progress_callback = progress_callback
                    self.log_callback = log_callback
                    self.total_epochs = total_epochs
                    self.epoch_start_time = None
                
                def on_epoch_begin(self, epoch, logs=None):
                    self.epoch_start_time = time.time()
                    if self.log_callback:
                        self.log_callback(f"🔄 Epoch {epoch + 1}/{self.total_epochs} starting...")
                
                def on_epoch_end(self, epoch, logs=None):
                    epoch_time = time.time() - (self.epoch_start_time or time.time())
                    progress = (epoch + 1) / self.total_epochs
                    
                    # Safely handle None logs
                    safe_logs = logs or {}
                    
                    if self.progress_callback:
                        self.progress_callback({
                            'epoch': epoch + 1,
                            'total_epochs': self.total_epochs,
                            'progress': progress,
                            'loss': safe_logs.get('loss', 0),
                            'accuracy': safe_logs.get('accuracy', 0),
                            'val_loss': safe_logs.get('val_loss', 0),
                            'val_accuracy': safe_logs.get('val_accuracy', 0),
                            'epoch_time': epoch_time
                        })
                    
                    if self.log_callback:
                        self.log_callback(
                            f"✅ Epoch {epoch + 1} completed in {epoch_time:.1f}s - "
                            f"Loss: {safe_logs.get('loss', 0):.4f}, "
                            f"Acc: {safe_logs.get('accuracy', 0):.4f}, "
                            f"Val_Acc: {safe_logs.get('val_accuracy', 0):.4f}"
                        )
            
            # Callbacks
            callbacks = [
                ProgressCallback(progress_callback, log_callback, epochs),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=2,
                    min_lr=1,
                    verbose=0
                )
            ]
            
            # Better steps calculation
            steps_per_epoch = max(20, min(200, train_gen.samples // train_gen.batch_size))
            validation_steps = max(10, min(100, val_gen.samples // val_gen.batch_size))
            
            if log_callback:
                log_callback("🎯 Starting training with enhanced settings...")
                log_callback(f"📊 Steps per epoch: {steps_per_epoch}")
                log_callback(f"🔍 Validation steps: {validation_steps}")
            
            # Train model
            history = model.fit(
                train_gen,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_data=val_gen,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose="auto",
                workers=1,
                use_multiprocessing=False
            )
            
            # Save model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{model_type}_enhanced_{timestamp}"
            model_path = self.models_dir / f"{model_name}.h5"
            
            model.save(model_path)
            
            # Get final metrics
            final_accuracy = 0.0
            final_val_accuracy = 0.0
            
            if history and hasattr(history, 'history'):
                if 'accuracy' in history.history and len(history.history['accuracy']) > 0:
                    final_accuracy = float(history.history['accuracy'][-1])
                if 'val_accuracy' in history.history and len(history.history['val_accuracy']) > 0:
                    final_val_accuracy = float(history.history['val_accuracy'][-1])
            
            # Save configuration
            config = {
                'class_names': class_names,  # This is the key fix
                'num_classes': num_classes,
                'model_type': model_type,
                'epochs': epochs,
                'actual_epochs': len(history.history['accuracy']) if history and hasattr(history, 'history') and 'accuracy' in history.history else 0,
                'timestamp': timestamp,
                'final_accuracy': final_accuracy,
                'final_val_accuracy': final_val_accuracy,
                'input_size': '128x128',
                'optimized': True,
                'learning_rate': learning_rate,
                'steps_per_epoch': steps_per_epoch,
                'validation_steps': validation_steps
            }
            
            config_path = str(model_path).replace('.h5', '_config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            if log_callback:
                log_callback(f"💾 Model saved: {model_name}.h5")
                log_callback(f"🎉 Training completed successfully!")
                log_callback(f"📊 Final accuracy: {final_accuracy:.4f}")
                log_callback(f"📊 Final validation accuracy: {final_val_accuracy:.4f}")
                log_callback(f"📂 Model can predict {len(class_names)} different classes")
                
                # Show class distribution for debugging
                class_counts = {}
                for class_name in class_names:
                    class_dir = Path(self.data_path) / class_name
                    if class_dir.exists():
                        count = len([f for f in class_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
                        class_counts[class_name] = count
                
                log_callback("📊 Dataset class distribution:")
                for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                    log_callback(f"   {class_name}: {count} images")
            
            self.training_active = False
            return True, model_name, history, config
            
        except Exception as e:
            self.training_active = False
            error_msg = f"Training failed: {str(e)}"
            if log_callback:
                log_callback(f"❌ {error_msg}")
            return False, None, None, str(e)

class PlantDiseaseDetector:
    def __init__(self):
        self.model = None
        self.class_names = []
        self.label_encoder = None
        self.input_size = (224, 224)  # Default size
        
        # PlantVillage dataset classes
        self.default_classes = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
            'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
            'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
            'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
            'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]
    
    def load_model(self, model_path):
        """Load trained model with automatic input size detection"""
        try:
            if not Path(model_path).exists():
                return False, f"Model not found: {model_path}"
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = tf.keras.models.load_model(model_path)
            
            # Try to load associated class names and config
            config_path = str(model_path).replace('.h5', '_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.class_names = config.get('class_names', self.default_classes)
                    # Check if model was trained with smaller input size
                    if config.get('input_size') == '128x128':
                        self.input_size = (128, 128)
            else:
                self.class_names = self.default_classes
            
            return True, "Model loaded successfully!"
        except Exception as e:
            return False, f"Error loading model: {str(e)[:100]}..."
    
    def preprocess_image(self, image):
        """Preprocess image for prediction with enhanced normalization"""
        try:
            # Convert to RGB array
            img_array = np.array(image.convert('RGB'))
            
            # Resize to model's expected input size
            img_resized = cv2.resize(img_array, self.input_size)
            
            # Normalize to [0, 1] range (same as training)
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            # Add batch dimension
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            return img_batch
        except Exception as e:
            st.error(f"Error preprocessing image: {e}")
            return None
    
    def parse_class_name(self, predicted_class):
        """Enhanced class name parsing with multiple fallback methods"""
        try:
            # Method 1: Standard PlantVillage format (Plant___Disease)
            if '___' in predicted_class:
                parts = predicted_class.split('___')
                plant_name = parts[0].replace('_', ' ').replace('(', ' (').title()
                condition = parts[1].replace('_', ' ').title()
                return plant_name, condition
            
            # Method 2: Single underscore format (Plant_Disease)  
            elif '_' in predicted_class:
                # Handle cases like "Apple_scab" or "Tomato_early_blight"
                parts = predicted_class.split('_')
                if len(parts) >= 2:
                    # First part is plant, rest is disease
                    plant_name = parts[0].title()
                    condition = ' '.join(parts[1:]).replace('_', ' ').title()
                    return plant_name, condition
            
            # Method 3: Space separated format
            elif ' ' in predicted_class:
                words = predicted_class.split()
                if len(words) >= 2:
                    plant_name = words[0].title()
                    condition = ' '.join(words[1:]).title()
                    return plant_name, condition
            
            # Method 4: Check if it's just a number (class index)
            if predicted_class.isdigit():
                return "Plant", f"Class {predicted_class}"
            
            # Method 5: Default fallback - treat as single word
            return predicted_class.title(), "Unknown Condition"
            
        except Exception as e:
            print(f"ERROR parsing class name: {e}")
            return "Unknown Plant", "Unknown Condition"

    def predict(self, image):
        """Make prediction with enhanced debugging and validation"""
        if self.model is None:
            return None, "Model not loaded"
        
        processed_image = self.preprocess_image(image)
        if processed_image is None:
            return None, "Error processing image"
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                predictions = self.model.predict(processed_image, verbose=0)
            
            # Get prediction probabilities
            prediction_probs = predictions[0]
            predicted_idx = np.argmax(prediction_probs)
            confidence = float(prediction_probs[predicted_idx])
            
            # Ensure we have enough classes
            if predicted_idx >= len(self.class_names):
                return None, f"Prediction index {predicted_idx} exceeds available classes ({len(self.class_names)})"
            
            predicted_class = self.class_names[predicted_idx]
            
            # Debug information
            print(f"DEBUG: Model output shape: {predictions.shape}")
            print(f"DEBUG: Number of classes: {len(self.class_names)}")
            print(f"DEBUG: Predicted index: {predicted_idx}")
            print(f"DEBUG: Predicted class: '{predicted_class}'")
            print(f"DEBUG: Confidence: {confidence:.4f}")
            print(f"DEBUG: Top 3 predictions:")
            top_3_indices = np.argsort(prediction_probs)[-3:][::-1]
            for i, idx in enumerate(top_3_indices):
                if idx < len(self.class_names):
                    print(f"  {i+1}. {self.class_names[idx]}: {prediction_probs[idx]:.4f}")
            
            # Enhanced class name parsing
            plant_name, condition = self.parse_class_name(predicted_class)
            
            # Determine if healthy
            is_healthy = any(word in condition.lower() for word in ['healthy', 'normal', 'good'])
            
            # Check if prediction is too confident (possible overfitting indicator)
            if confidence > 0.999:
                print("WARNING: Very high confidence detected - possible overfitting")
            
            # Get all predictions for analysis
            all_predictions = {}
            for i, prob in enumerate(prediction_probs):
                if i < len(self.class_names):
                    all_predictions[self.class_names[i]] = float(prob)
            
            return {
                'class': predicted_class,
                'confidence': confidence,
                'plant': plant_name,
                'condition': condition,
                'is_healthy': is_healthy,
                'all_predictions': all_predictions,
                'debug_info': {
                    'predicted_idx': predicted_idx,
                    'num_classes': len(self.class_names),
                    'input_size': self.input_size,
                    'model_output_shape': predictions.shape
                }
            }, "Success"
        except Exception as e:
            return None, f"Prediction error: {str(e)}"

def validate_dataset():
    """Check if dataset exists and validate structure"""
    if not DATA_DIR.exists():
        return False, f"Dataset directory not found: {DATA_DIR}"
    
    subdirs = [d for d in DATA_DIR.iterdir() if d.is_dir()]
    if len(subdirs) == 0:
        return False, "No class directories found in dataset"
    
    total_images = 0
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    
    for subdir in subdirs:
        images = [f for f in subdir.iterdir() if f.suffix.lower() in valid_extensions]
        total_images += len(images)
    
    return True, f"Found {len(subdirs)} classes with {total_images} images"

# Initialize app state
if 'app_initialized' not in st.session_state:
    st.session_state.app_initialized = True
    st.session_state.detector = PlantDiseaseDetector()
    st.session_state.model_loaded = False
    st.session_state.training_status = 'idle'
    st.session_state.training_progress = {}
    st.session_state.training_logs = []
    st.session_state.trainer = None

# Get detector from session state
detector = st.session_state.detector

# Header
st.markdown('<h1 class="main-header">🌿 Plant Disease Detection System</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; font-size: 1.3rem; color: #666; margin-bottom: 3rem; font-family: 'Inter', sans-serif;">
    <strong>AI-powered plant health analysis for sustainable agriculture</strong><br>
    <span style="font-size: 1rem; color: #999;">Upload plant images to detect diseases with advanced deep learning</span>
</div>
""", unsafe_allow_html=True)

# Show module status if some are missing
if not LOCAL_MODULES_AVAILABLE:
    st.warning("⚠️ Some advanced features may not be available due to missing modules. Basic functionality is still working.")

# Status metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    model_count = len(list(MODELS_DIR.glob("*.h5"))) if MODELS_DIR.exists() else 0
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color: #2E7D32; margin: 0; font-size: 1.1rem;">🤖 AI Models</h3>
        <p style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0; color: #1B5E20;">{model_count}</p>
        <p style="color: #666; margin: 0; font-size: 0.9rem;">Available</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    dataset_valid, dataset_msg = validate_dataset()
    status_icon = "✅" if dataset_valid else "❌"
    status_text = "Ready" if dataset_valid else "Missing"
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color: #1976D2; margin: 0; font-size: 1.1rem;">📊 Dataset</h3>
        <p style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0; color: #0D47A1;">{status_icon}</p>
        <p style="color: #666; margin: 0; font-size: 0.9rem;">{status_text}</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    model_status = "Online" if st.session_state.model_loaded else "Load Model"
    status_color = "#2E7D32" if st.session_state.model_loaded else "#FF9800"
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color: #7B1FA2; margin: 0; font-size: 1.1rem;">🎯 Status</h3>
        <p style="font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0; color: {status_color};">{model_status}</p>
        <p style="color: #666; margin: 0; font-size: 0.9rem;">System</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    training_status = st.session_state.training_status.title()
    training_color = "#FF9800" if st.session_state.training_status == 'training' else "#4CAF50"
    st.markdown(f"""
    <div class="metric-card {'training-active' if st.session_state.training_status == 'training' else ''}">
        <h3 style="color: #D32F2F; margin: 0; font-size: 1.1rem;">⚡ Training</h3>
        <p style="font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0; color: {training_color};">{training_status}</p>
        <p style="color: #666; margin: 0; font-size: 0.9rem;">Available</p>
    </div>
    """, unsafe_allow_html=True)

# Main tabs
tab1, tab2, tab3 = st.tabs(["🔍 **Disease Detection**", "🔧 **Train Model**", "📊 **System Info**"])

with tab1:
    # Sidebar - Model Selection
    with st.sidebar:
        st.markdown('<h2 class="sub-header">🤖 Model Management</h2>', unsafe_allow_html=True)
        
        # Available models
        model_files = list(MODELS_DIR.glob("*.h5")) if MODELS_DIR.exists() else []
        
        if not model_files:
            st.markdown("""
            <div class="warning-card">
                <h4>⚠️ No Models Found</h4>
                <p>No trained models available. Please train a model first using the 'Train Model' tab.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            model_names = [f.stem for f in model_files]
            selected_model = st.selectbox(
                "Choose Model",
                model_names,
                help="Select a trained model for prediction"
            )
            
            if st.button("🔄 Load Model", type="primary", use_container_width=True):
                model_path = MODELS_DIR / f"{selected_model}.h5"
                
                with st.spinner("Loading model..."):
                    success, message = detector.load_model(model_path)
                    
                    if success:
                        st.success(message)
                        st.session_state.model_loaded = True
                    else:
                        st.error(message)
                        st.session_state.model_loaded = False
            
            # Model info
            if st.session_state.model_loaded:
                st.markdown("""
                <div class="info-card">
                    <h4>✅ Model Ready</h4>
                    <p><strong>Classes:</strong> {}</p>
                    <p><strong>Input:</strong> {}×{} RGB</p>
                    <p><strong>Architecture:</strong> Deep CNN</p>
                </div>
                """.format(len(detector.class_names), detector.input_size[0], detector.input_size[1]), unsafe_allow_html=True)
                
                # Debug section - Add this to see actual class names
                with st.expander("🔍 Debug: View Model Classes"):
                    st.write("**Model's actual class names:**")
                    for i, class_name in enumerate(detector.class_names[:10]):  # Show first 10
                        st.write(f"{i}: `{class_name}`")
                    if len(detector.class_names) > 10:
                        st.write(f"... and {len(detector.class_names) - 10} more classes")
    
    # Main prediction area
    if not st.session_state.model_loaded:
        st.markdown("""
        <div class="info-card">
            <h3>🤖 Welcome to Plant Disease Detection</h3>
            <p>This AI-powered system helps identify plant diseases from leaf images.</p>
            <p><strong>To get started:</strong></p>
            <ol>
                <li>Train a model using the 'Train Model' tab, or</li>
                <li>Load a pre-trained model from the sidebar (if available)</li>
                <li>Upload a plant image for analysis</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Show demo section
        st.markdown('<h2 class="sub-header">📤 Upload Plant Image (Demo Mode)</h2>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a plant leaf image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a plant leaf for disease analysis"
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 📷 Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, caption="Plant Image", use_column_width=True)
                
                # Image details
                st.markdown(f"""
                <div class="info-card">
                    <h4>📋 Image Details</h4>
                    <p><strong>Size:</strong> {image.size[0]} × {image.size[1]} pixels</p>
                    <p><strong>Format:</strong> {image.format}</p>
                    <p><strong>Mode:</strong> {image.mode}</p>
                    <p><strong>File Size:</strong> {len(uploaded_file.getvalue()) / 1024:.1f} KB</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 🔍 Analysis Results")
                st.markdown("""
                <div class="warning-card">
                    <h4>⚠️ Demo Mode</h4>
                    <p>No model is currently loaded. Please train or load a model to get predictions.</p>
                    <p>Your image has been successfully uploaded and is ready for analysis once a model is available.</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        # Full prediction interface when model is loaded
        st.markdown('<h2 class="sub-header">📤 Upload Plant Image</h2>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a plant leaf image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a plant leaf for disease analysis"
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns([1, 1.3])
            
            with col1:
                st.markdown("### 📷 Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, caption="Plant Image", use_column_width=True)
                
                # Image details
                st.markdown(f"""
                <div class="info-card">
                    <h4>📋 Image Details</h4>
                    <p><strong>Size:</strong> {image.size[0]} × {image.size[1]} pixels</p>
                    <p><strong>Format:</strong> {image.format}</p>
                    <p><strong>Mode:</strong> {image.mode}</p>
                    <p><strong>File Size:</strong> {len(uploaded_file.getvalue()) / 1024:.1f} KB</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 🔍 Analysis Results")
                
                with st.spinner("🧠 Analyzing image with AI..."):
                    result, status = detector.predict(image)
                
                if result is None:
                    st.markdown(f"""
                    <div class="error-card">
                        <h4>❌ Analysis Failed</h4>
                        <p>{status}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Main prediction
                    if result['is_healthy']:
                        st.markdown(f"""
                        <div class="prediction-card healthy-card">
                            <h3>✅ Healthy Plant Detected</h3>
                            <div style="margin: 1rem 0;">
                                <h4>🌱 Plant: <span style="color: #2E7D32;">{result['plant']}</span></h4>
                                <h4>🍃 Condition: <span style="color: #2E7D32;">{result['condition']}</span></h4>
                                <h4>🎯 Confidence: <span style="color: #2E7D32;">{result['confidence']:.1%}</span></h4>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.success("🎉 Excellent! Your plant appears to be healthy and thriving.")
                    else:
                        st.markdown(f"""
                        <div class="prediction-card diseased-card">
                            <h3>⚠️ Disease Detected</h3>
                            <div style="margin: 1rem 0;">
                                <h4>🌱 Plant: <span style="color: #D32F2F;">{result['plant']}</span></h4>
                                <h4>🦠 Disease: <span style="color: #D32F2F;">{result['condition']}</span></h4>
                                <h4>🎯 Confidence: <span style="color: #D32F2F;">{result['confidence']:.1%}</span></h4>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.error("🚨 Disease detected! Consider consulting an agricultural expert for treatment recommendations.")
                    
                    # Confidence visualization
                    st.markdown("#### 📊 Confidence Score")
                    st.progress(result['confidence'])
                    
                    # Additional info based on confidence
                    if result['confidence'] > 0.9:
                        st.info("🎯 Very high confidence - Reliable prediction")
                    elif result['confidence'] > 0.7:
                        st.info("✅ Good confidence - Consider additional analysis")
                    else:
                        st.warning("⚠️ Low confidence - Consider retaking image or consulting expert")
                    
                    # Top predictions
                    with st.expander("🔍 View All Predictions"):
                        top_predictions = sorted(
                            result['all_predictions'].items(),
                            key=lambda x: x[1],
                            reverse=True
                        )[:5]
                        
                        for i, (class_name, confidence) in enumerate(top_predictions):
                            parts = class_name.split('___')
                            plant = parts[0].replace('_', ' ').title() if len(parts) > 0 else "Unknown"
                            condition = parts[1].replace('_', ' ').title() if len(parts) > 1 else "Unknown"
                            rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i]
                            st.write(f"{rank_emoji} **{plant} - {condition}:** {confidence:.1%}")

with tab2:
    st.markdown('<h2 class="sub-header">🔧 Train New Model</h2>', unsafe_allow_html=True)
    
    # Dataset validation
    dataset_valid, dataset_msg = validate_dataset()
    
    if not dataset_valid:
        st.markdown(f"""
        <div class="error-card">
            <h3>❌ Dataset Not Found</h3>
            <p>{dataset_msg}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### 📁 Dataset Setup Instructions
        
        **Expected directory structure:**
        ```
        data/
        └── plantvillage/
            ├── Apple___Apple_scab/
            │   ├── image1.jpg
            │   ├── image2.jpg
            │   └── ...
            ├── Apple___Black_rot/
            ├── Apple___healthy/
            └── ... (other plant disease folders)
        ```
        
        **Setup steps:**
        1. 📥 Download the PlantVillage dataset
        2. 📂 Create directory: `data/plantvillage/`
        3. 🗂️ Extract disease folders into `data/plantvillage/`
        4. ✅ Ensure each disease has its own folder with images
        """)
    else:
        st.markdown(f"""
        <div class="success-card">
            <h4>✅ Dataset Validated</h4>
            <p>{dataset_msg}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Training Configuration
        st.markdown("### ⚙️ Training Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Model Architecture",
                ["ultra_fast", "efficient", "transfer"],
                format_func=lambda x: {
                    "ultra_fast": "Ultra Fast CNN (Demo - 1-2 min)",
                    "efficient": "Efficient CNN (Fast - 3-5 min)", 
                    "transfer": "Transfer Learning (Moderate - 5-10 min)"
                }[x],
                help="Choose model type based on your time constraints"
            )
            
            epochs = st.slider(
                "Training Epochs", 
                min_value=3, 
                max_value=20, 
                value=5 if model_type == "ultra_fast" else 10,
                help="Fewer epochs = faster training"
            )
        
        with col2:
            st.markdown("#### ⚡ Speed Optimizations")
            st.info(f"**Ultra Fast**: ~{epochs * 1:.1f} minutes (128x128 images)")
            st.info(f"**Efficient**: ~{epochs * 2:.1f} minutes (128x128 images)")  
            st.info(f"**Transfer**: ~{epochs * 3:.1f} minutes (128x128 images)")
            
            if model_type == "ultra_fast":
                st.warning("⚡ **Ultra Fast mode**: Optimized for demo purposes. Training will complete all epochs.")
            
            # Add training tips
            with st.expander("💡 Training Tips"):
                st.markdown("""
                **If training stops early:**
                - Use **Ultra Fast** mode for guaranteed completion
                - Increase epochs (more epochs = less likely to stop early)
                - Check dataset has enough samples per class
                
                **For best results:**
                - Start with 5-10 epochs for testing
                - Gradually increase epochs for better accuracy
                - Monitor validation accuracy trends
                """)
        
        # Training Controls
        st.markdown("### 🎮 Training Controls")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Start Training", type="primary", use_container_width=True, disabled=(st.session_state.training_status == 'training')):
                if st.session_state.training_status != 'training':
                    # Initialize trainer
                    st.session_state.trainer = FastModelTrainer(DATA_DIR, MODELS_DIR)
                    st.session_state.training_status = 'training'
                    st.session_state.training_logs = []
                    st.session_state.training_progress = {}
                    
                    # Start training in a separate thread (simulated for demo)
                    st.rerun()
        
        with col2:
            if st.button("⏹️ Stop Training", use_container_width=True):
                if st.session_state.training_status == 'training':
                    st.session_state.training_status = 'stopped'
                    st.warning("Training stopped by user")
        
        with col3:
            if st.button("🗑️ Clear Logs", use_container_width=True):
                st.session_state.training_logs = []
                st.session_state.training_progress = {}
                st.success("Logs cleared")
        
        # Training Progress Section
        if st.session_state.training_status == 'training':
            # Initialize trainer if not exists
            if st.session_state.trainer is None:
                st.session_state.trainer = FastModelTrainer(DATA_DIR, MODELS_DIR)
            
            # Progress callback functions
            def update_progress(progress_data):
                st.session_state.training_progress = progress_data
            
            def add_log(message):
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.training_logs.append(f"[{timestamp}] {message}")
                # Keep only last 20 logs
                st.session_state.training_logs = st.session_state.training_logs[-20:]
            
            # Start actual training
            with st.spinner("Initializing training..."):
                success, model_name, history, config = st.session_state.trainer.train_model_with_progress(
                    model_type=model_type,
                    epochs=epochs,
                    progress_callback=update_progress,
                    log_callback=add_log
                )
            
            if success:
                st.session_state.training_status = 'completed'
                st.success(f"🎉 Training completed! Model saved as: {model_name}")
                
                # Show final results
                st.markdown(f"""
                <div class="success-card">
                    <h4>🎯 Training Results</h4>
                    <p><strong>Model Name:</strong> {model_name}</p>
                    <p><strong>Final Accuracy:</strong> {config['final_accuracy']:.4f}</p>
                    <p><strong>Final Validation Accuracy:</strong> {config['final_val_accuracy']:.4f}</p>
                    <p><strong>Classes:</strong> {config['num_classes']}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.session_state.training_status = 'failed'
                st.error(f"Training failed: {config}")
        
        # Display training progress
        if st.session_state.training_progress:
            progress_data = st.session_state.training_progress
            
            st.markdown("""
            <div class="progress-card">
                <h4>📊 Training Progress</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress bar
            progress_value = progress_data.get('progress', 0)
            st.progress(progress_value)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Epoch", 
                    f"{progress_data.get('epoch', 0)}/{progress_data.get('total_epochs', 0)}"
                )
            
            with col2:
                loss = progress_data.get('loss', 0)
                st.metric("Loss", f"{loss:.4f}")
            
            with col3:
                acc = progress_data.get('accuracy', 0)
                st.metric("Accuracy", f"{acc:.1%}")
            
            with col4:
                val_acc = progress_data.get('val_accuracy', 0)
                st.metric("Val Accuracy", f"{val_acc:.1%}")
            
            # Time info
            if 'epoch_time' in progress_data:
                remaining_epochs = progress_data.get('total_epochs', 0) - progress_data.get('epoch', 0)
                estimated_time = remaining_epochs * progress_data.get('epoch_time', 0)
                st.info(f"⏱️ Epoch time: {progress_data['epoch_time']:.1f}s | Estimated remaining: {estimated_time/60:.1f} minutes")
        
        # Training logs
        if st.session_state.training_logs:
            st.markdown("### 📝 Training Logs")
            log_text = "\n".join(st.session_state.training_logs)
            st.markdown(f'<div class="training-log">{log_text}</div>', unsafe_allow_html=True)
        
        # Training not available fallback
        if not dataset_valid:
            st.markdown("""
            <div class="warning-card">
                <h3>⚠️ Training Not Available</h3>
                <p>Please set up the dataset first to enable model training.</p>
            </div>
            """, unsafe_allow_html=True)

with tab3:
    st.markdown('<h2 class="sub-header">📊 System Information</h2>', unsafe_allow_html=True)
    
    # Available models
    st.markdown("### 🤖 Available Models")
    model_files = list(MODELS_DIR.glob("*.h5")) if MODELS_DIR.exists() else []
    
    if model_files:
        for model_file in model_files:
            file_size = model_file.stat().st_size / (1024 * 1024)
            mod_time = time.ctime(model_file.stat().st_mtime)
            
            # Try to load config
            config_path = str(model_file).replace('.h5', '_config.json')
            config_info = ""
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        config_info = f"<p><strong>Accuracy:</strong> {config.get('final_val_accuracy', 'N/A'):.4f}</p>"
                except:
                    config_info = ""
            
            st.markdown(f"""
            <div class="info-card">
                <h4>📁 {model_file.stem}</h4>
                <p><strong>Size:</strong> {file_size:.1f} MB</p>
                <p><strong>Modified:</strong> {mod_time}</p>
                <p><strong>Type:</strong> {'Transfer Learning' if 'transfer' in model_file.stem else 'Efficient CNN'}</p>
                {config_info}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-card">
            <h4>⚠️ No Models Found</h4>
            <p>No trained models available. Use the 'Train Model' tab to create your first model.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # System specifications
    st.markdown("### 🖥️ System Specifications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="info-card">
            <h4>🐍 Python Environment</h4>
            <p><strong>TensorFlow:</strong> {tf.__version__}</p>
            <p><strong>Python:</strong> {sys.version.split()[0]}</p>
            <p><strong>NumPy:</strong> {np.__version__}</p>
            <p><strong>Streamlit:</strong> {st.__version__}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        try:
            gpus = tf.config.list_physical_devices('GPU')
            gpu_info = f"{len(gpus)} GPU(s) available" if gpus else "CPU only"
        except:
            gpu_info = "CPU only"
        
        dataset_valid, _ = validate_dataset()
        dataset_status = "✅ Ready" if dataset_valid else "❌ Missing"
        
        st.markdown(f"""
        <div class="info-card">
            <h4>⚡ Hardware & Status</h4>
            <p><strong>Hardware:</strong> {gpu_info}</p>
            <p><strong>Dataset:</strong> {dataset_status}</p>
            <p><strong>Training:</strong> {'✅ Ready' if dataset_valid else '❌ Dataset Required'}</p>
            <p><strong>Models Dir:</strong> {'✅ Exists' if MODELS_DIR.exists() else '❌ Missing'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance recommendations
    st.markdown("### 💡 Quick Start Guide")
    
    if not dataset_valid:
        st.error("📁 **Step 1**: Set up the PlantVillage dataset in `data/plantvillage/`")
    else:
        st.success("📁 **Step 1**: Dataset is ready ✅")
    
    if len(model_files) == 0:
        st.warning("🤖 **Step 2**: Train your first model using the 'Train Model' tab")
    else:
        st.success(f"🤖 **Step 2**: {len(model_files)} model(s) available ✅")
    
    st.info("🔍 **Step 3**: Upload plant images in 'Disease Detection' tab for analysis")
    
    # Training optimization tips
    with st.expander("🚀 Training Optimization Tips"):
        st.markdown("""
        **For Fast Training:**
        - Use "Efficient CNN" model type
        - Start with 10-15 epochs
        - Monitor validation accuracy to avoid overfitting
        
        **For Best Accuracy:**
        - Use "Transfer Learning" with MobileNetV2
        - Train for 20-30 epochs
        - Use GPU if available
        
        **Memory Optimization:**
        - Reduce batch size if running out of memory
        - Close other applications during training
        - Use CPU training for very large datasets
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem;">
    <p>🌿 Plant Disease Detection System | Made with ❤️ using Streamlit and TensorFlow</p>
    <p>Upload plant images • Train AI models • Detect diseases • Improve crop health</p>
</div>
""", unsafe_allow_html=True)

def debug_model_classes(model_path):
    """Debug function to check model's class names"""
    config_path = str(model_path).replace('.h5', '_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            st.write("**Model's actual class names:**")
            for i, class_name in enumerate(config.get('class_names', [])):
                st.write(f"{i}: {class_name}")

# Add this to your streamlit app for easy testing
def show_test_image_examples():
    st.markdown("### 📸 Test Image Examples")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🍎 Apple Diseases:**
        - Apple scab (brown spots)
        - Apple black rot (dark lesions)
        - Cedar apple rust (orange spots)
        - Healthy apple leaves
        """)
    
    with col2:
        st.markdown("""
        **🍅 Tomato Diseases:**
        - Early blight (dark spots with rings)
        - Late blight (water-soaked lesions)
        - Leaf mold (yellow patches)
        - Healthy tomato leaves
        """)
    
    with col3:
        st.markdown("""
        **🥔 Potato Diseases:**
        - Early blight (brown lesions)
        - Late blight (dark patches)
        - Healthy potato leaves
        """)

# Add this in your Disease Detection tab
if not st.session_state.model_loaded:
    show_test_image_examples()

# Add this helper function to generate test URLs
def get_test_image_urls():
    return {
        "Apple Scab": "https://example.com/apple_scab.jpg",
        "Tomato Healthy": "https://example.com/tomato_healthy.jpg", 
        "Potato Blight": "https://example.com/potato_blight.jpg"
    }

# Or create a simple image search helper
st.markdown("""
### 🔍 **Where to find test images:**
1. **Kaggle PlantVillage Dataset**: Most reliable source
2. **Google Images**: Search for specific diseases
3. **Agricultural websites**: University extension services
4. **Plant pathology databases**: Research institutions

**Search terms to try:**
- "apple scab leaf disease"
- "tomato early blight"  
- "healthy potato leaf"
- "corn rust disease"
- "grape black rot"
""")