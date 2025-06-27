import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from pathlib import Path
from keras.utils import to_categorical

class DataPreprocessor:
    def __init__(self, data_path, img_size=(224, 224), batch_size=32):
        self.data_path = data_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.label_encoder = LabelEncoder()
        
    def load_dataset(self):
        """Load and organize the PlantVillage dataset"""
        # Add data path validation
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data directory not found: {self.data_path}")
        
        if not os.path.isdir(self.data_path):
            raise ValueError(f"Data path is not a directory: {self.data_path}")
    
        print(f"Loading data from: {os.path.abspath(self.data_path)}")
    
        images = []
        labels = []
        total_files = 0
    
        # Walk through the dataset directory
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                total_files += 1
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Extract label from folder name
                    label = os.path.basename(root)
                    image_path = os.path.join(root, file)
                
                    # Load and preprocess image
                    image = self.load_and_preprocess_image(image_path)
                    if image is not None:
                        images.append(image)
                        labels.append(label)

        print(f"Found {total_files} total files")
        print(f"Loaded {len(images)} valid images with {len(set(labels))} classes")
    
        if len(images) == 0:
            raise ValueError("No valid images found in the dataset directory")
        
        return np.array(images), np.array(labels)
    
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess a single image"""
        try:
            # Read image
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize image
            img = cv2.resize(img, self.img_size)
            
            # Normalize pixel values
            img = img.astype(np.float32) / 255.0
            
            return img
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def create_data_generators(self, X_train, y_train, X_val, y_val):
        """Create data generators with augmentation"""
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        # No augmentation for validation
        val_datagen = ImageDataGenerator()
        
        train_generator = train_datagen.flow(
            X_train, y_train,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        val_generator = val_datagen.flow(
            X_val, y_val,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def prepare_data(self, test_size=0.2, val_size=0.1):
        """Complete data preparation pipeline"""
        # Load dataset
        X, y = self.load_dataset()
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        y_categorical = to_categorical(y_encoded)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_categorical, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Create data generators
        train_gen, val_gen = self.create_data_generators(X_train, y_train, X_val, y_val)
        
        return {
            'train_generator': train_gen,
            'val_generator': val_gen,
            'X_test': X_test,
            'y_test': y_test,
            'label_encoder': self.label_encoder,
            'num_classes': len(self.label_encoder.classes_)
        }
    
    def visualize_samples(self, X, y, num_samples=12):
        """Visualize sample images with labels"""
        fig, axes = plt.subplots(3, 4, figsize=(15, 12))
        axes = axes.ravel()
        
        for i in range(min(num_samples, len(X))):
            axes[i].imshow(X[i])
            if len(y.shape) > 1:  # One-hot encoded
                label_idx = np.argmax(y[i])
                label = self.label_encoder.inverse_transform([label_idx])[0]
            else:
                label = y[i]
            axes[i].set_title(f'Class: {label}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()