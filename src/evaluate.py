import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf
from keras.models import load_model

class ModelEvaluator:
    def __init__(self, model_path, label_encoder):
        self.model = load_model(model_path)
        self.label_encoder = label_encoder
        self.class_names = label_encoder.classes_
        
    def evaluate_model(self, X_test, y_test):
        """Comprehensive model evaluation"""
        # Check if model is loaded
        if self.model is None:
            raise ValueError("Model failed to load. Please check the model path and ensure the model exists.")
        # Make predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
        # Plot confusion matrix
        self.plot_confusion_matrix(y_true, y_pred)
        
        # Visualize predictions
        self.visualize_predictions(X_test, y_test, y_pred_proba, num_samples=12)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def visualize_predictions(self, X_test, y_test, y_pred_proba, num_samples=12):
        """Visualize sample predictions"""
        indices = np.random.choice(len(X_test), num_samples, replace=False)
        
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            # Display image
            axes[i].imshow(X_test[idx])
            
            # Get true and predicted labels
            true_label = self.class_names[np.argmax(y_test[idx])]
            pred_label = self.class_names[np.argmax(y_pred_proba[idx])]
            confidence = np.max(y_pred_proba[idx])
            
            # Set title with colors
            color = 'green' if true_label == pred_label else 'red'
            axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}',
                            color=color, fontsize=10)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_class_distribution(self, y_test):
        """Plot class distribution in test set"""
        y_true = np.argmax(y_test, axis=1)
        class_counts = np.bincount(y_true)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(self.class_names, class_counts)
        plt.title('Class Distribution in Test Set')
        plt.xlabel('Classes')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, class_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def predict_single_image(self, image):
        """Predict class for a single image"""
        if self.model is None:
            raise ValueError("Model is not loaded. Please check the model path and ensure the model exists.")
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        prediction = self.model.predict(image)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        return {
            'class': self.class_names[predicted_class],
            'confidence': confidence,
            'probabilities': dict(zip(self.class_names, prediction[0]))
        }