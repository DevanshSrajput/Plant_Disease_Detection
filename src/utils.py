import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def create_directory_structure():
    """Create necessary directories for the project"""
    directories = [
        'data/plantvillage',
        'models',
        'results',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("Directory structure created successfully!")

def load_and_preprocess_single_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single image for prediction"""
    try:
        # Read image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img = cv2.resize(img, target_size)
        
        # Normalize pixel values
        img = img.astype(np.float32) / 255.0
        
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def save_model_config(config, filepath):
    """Save model configuration to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)

def load_model_config(filepath):
    """Load model configuration from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_model_size(model):
    """Calculate model size in MB"""
    param_count = model.count_params()
    # Assuming float32 parameters (4 bytes each)
    size_mb = (param_count * 4) / (1024 * 1024)
    return size_mb

def get_class_weights(y_train):
    """Calculate class weights for imbalanced datasets"""
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(np.argmax(y_train, axis=1))
    class_weights = compute_class_weight(
        'balanced',
        classes=classes,
        y=np.argmax(y_train, axis=1)
    )
    
    return dict(zip(classes, class_weights))

def plot_sample_augmentations(datagen, image, num_samples=6):
    """Visualize data augmentation effects"""
    image_batch = np.expand_dims(image, axis=0)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Augmented images
    i = 1
    for batch in datagen.flow(image_batch, batch_size=1):
        if i >= num_samples:
            break
        axes[i].imshow(batch[0])
        axes[i].set_title(f'Augmented {i}')
        axes[i].axis('off')
        i += 1
    
    plt.tight_layout()
    plt.show()

def monitor_gpu_usage():
    """Monitor GPU usage and memory"""
    import tensorflow as tf
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU acceleration is available. Found {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPU found. Using CPU.")

def create_dataset_summary(data_path):
    """Create a summary of the dataset"""
    summary = {
        'total_images': 0,
        'classes': {},
        'image_formats': {},
    }
    
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                summary['total_images'] += 1
                
                # Count by class (folder name)
                class_name = os.path.basename(root)
                summary['classes'][class_name] = summary['classes'].get(class_name, 0) + 1
                
                # Count by format
                ext = os.path.splitext(file)[1].lower()
                summary['image_formats'][ext] = summary['image_formats'].get(ext, 0) + 1
    
    return summary

def print_dataset_info(summary):
    """Print dataset information"""
    print("=== Dataset Summary ===")
    print(f"Total Images: {summary['total_images']}")
    print(f"Number of Classes: {len(summary['classes'])}")
    print("\nClass Distribution:")
    for class_name, count in summary['classes'].items():
        print(f"  {class_name}: {count} images")
    
    print("\nImage Formats:")
    for format_name, count in summary['image_formats'].items():
        print(f"  {format_name}: {count} images")