import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PLANTVILLAGE_DIR, MODELS_DIR, validate_data_directory
import numpy as np
import matplotlib.pyplot as plt
from src.data_preprocessing import DataPreprocessor
from src.model import PlantDiseaseModel
import tensorflow as tf

class ModelTrainer:
    def __init__(self, data_path=None, model_save_path=None):
        self.data_path = data_path or str(PLANTVILLAGE_DIR)
        self.model_save_path = model_save_path or str(MODELS_DIR)
        
        # Validate paths
        if not validate_data_directory():
            raise FileNotFoundError("Data directory validation failed")
            
        os.makedirs(self.model_save_path, exist_ok=True)
        
    def train_model(self, model_type='custom', epochs=50, base_model_name='ResNet50'):
        """Complete training pipeline"""
        
        # Prepare data
        print(f"Preparing data from: {self.data_path}")
        preprocessor = DataPreprocessor(self.data_path)
        data = preprocessor.prepare_data()
        
        # Create model
        print(f"Creating {model_type} model...")
        model_builder = PlantDiseaseModel(num_classes=data['num_classes'])
        
        if model_type == 'custom':
            model = model_builder.create_custom_cnn()
        else:
            model = model_builder.create_transfer_learning_model(base_model_name)
        
        # Compile model
        model = model_builder.compile_model(model)
        
        # Print model summary
        print("Model Architecture:")
        model.summary()
        
        # Get callbacks
        callbacks = model_builder.get_callbacks(
            os.path.join(self.model_save_path, f'{model_type}_best_model.h5')
        )
        
        # Train model
        print("Starting training...")
        history = model.fit(
            data['train_generator'],
            epochs=epochs,
            validation_data=data['val_generator'],
            callbacks=callbacks,
            verbose="auto"
        )
        
        # Save final model
        model.save(os.path.join(self.model_save_path, f'{model_type}_final_model.h5'))
        
        # Plot training history
        self.plot_training_history(history, model_type)
        
        return model, history, data
    
    def plot_training_history(self, history, model_type):
        """Plot training and validation metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title(f'{model_type} Model - Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Training Loss')
        axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title(f'{model_type} Model - Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Top-3 Accuracy (if available)
        if 'top_3_accuracy' in history.history:
            axes[1, 0].plot(history.history['top_3_accuracy'], label='Training Top-3 Accuracy')
            axes[1, 0].plot(history.history['val_top_3_accuracy'], label='Validation Top-3 Accuracy')
            axes[1, 0].set_title(f'{model_type} Model - Top-3 Accuracy')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Top-3 Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Learning rate (if available)
        if 'lr' in history.history:
            axes[1, 1].plot(history.history['lr'], label='Learning Rate')
            axes[1, 1].set_title(f'{model_type} Model - Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_save_path, f'{model_type}_training_history.png'))
        plt.show()

if __name__ == "__main__":
    # Example usage
    trainer = ModelTrainer('data/plantvillage')
    
    # Train custom CNN
    print("Training Custom CNN...")
    custom_model, custom_history, data = trainer.train_model('custom', epochs=30)
    
    # Train transfer learning model
    print("\nTraining Transfer Learning Model...")
    transfer_model, transfer_history, _ = trainer.train_model('transfer', epochs=20, base_model_name='ResNet50')