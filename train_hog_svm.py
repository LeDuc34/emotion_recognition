import numpy as np
import cv2
import os
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from skimage.feature import hog
import pickle
from tqdm import tqdm

# Emotion labels for FER-2013
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
EMOTION_MAP = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'sad': 4,
    'surprise': 5,
    'neutral': 6
}

class HOGSVMTrainer:
    """Trainer for HOG + Linear SVM classifier"""
    def __init__(self):
        self.hog_params = {
            'orientations': 9,
            'pixels_per_cell': (8, 8),
            'cells_per_block': (2, 2),
            'block_norm': 'L2-Hys'
        }
        self.model = None
    
    def extract_hog_features(self, image):
        """Extract HOG features from grayscale image (48x48)"""
        # Si l'image n'est pas déjà en 48x48, la redimensionner
        if image.shape != (48, 48):
            image = cv2.resize(image, (48, 48))
        
        features = hog(image, 
                      orientations=self.hog_params['orientations'],
                      pixels_per_cell=self.hog_params['pixels_per_cell'],
                      cells_per_block=self.hog_params['cells_per_block'],
                      block_norm=self.hog_params['block_norm'],
                      visualize=False)
        return features
    
    def load_fer2013_from_folders(self, base_path='archive'):
        """
        Load FER-2013 dataset from folder structure:
        archive/
          train/
            angry/
            disgust/
            fear/
            happy/
            sad/
            surprise/
            neutral/
          test/
            angry/
            ...
        """
        print("Loading FER-2013 dataset from folders...")
        
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        
        # Charger les données d'entraînement
        train_path = os.path.join(base_path, 'train')
        if os.path.exists(train_path):
            print("\nLoading training data...")
            for emotion in EMOTIONS:
                emotion_path = os.path.join(train_path, emotion)
                if not os.path.exists(emotion_path):
                    print(f"⚠ Warning: {emotion_path} not found")
                    continue
                
                emotion_label = EMOTION_MAP[emotion]
                image_files = [f for f in os.listdir(emotion_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
                
                print(f"  Loading {emotion}: {len(image_files)} images")
                for img_file in tqdm(image_files, desc=f"  {emotion}", leave=False):
                    img_path = os.path.join(emotion_path, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is not None:
                        # Redimensionner à 48x48 si nécessaire
                        if img.shape != (48, 48):
                            img = cv2.resize(img, (48, 48))
                        
                        X_train.append(img.flatten())
                        y_train.append(emotion_label)
        
        # Charger les données de test
        test_path = os.path.join(base_path, 'test')
        if os.path.exists(test_path):
            print("\nLoading test data...")
            for emotion in EMOTIONS:
                emotion_path = os.path.join(test_path, emotion)
                if not os.path.exists(emotion_path):
                    print(f"⚠ Warning: {emotion_path} not found")
                    continue
                
                emotion_label = EMOTION_MAP[emotion]
                image_files = [f for f in os.listdir(emotion_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
                
                print(f"  Loading {emotion}: {len(image_files)} images")
                for img_file in tqdm(image_files, desc=f"  {emotion}", leave=False):
                    img_path = os.path.join(emotion_path, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is not None:
                        # Redimensionner à 48x48 si nécessaire
                        if img.shape != (48, 48):
                            img = cv2.resize(img, (48, 48))
                        
                        X_test.append(img.flatten())
                        y_test.append(emotion_label)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        print(f"\nDataset loaded:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"\nTraining class distribution:")
        for emotion_idx, emotion_name in enumerate(EMOTIONS):
            count = np.sum(y_train == emotion_idx)
            percentage = (count / len(y_train)) * 100 if len(y_train) > 0 else 0
            print(f"  {emotion_idx} - {emotion_name:10s}: {count:5d} ({percentage:5.2f}%)")
        
        return X_train, y_train, X_test, y_test
    
    def extract_features_batch(self, X):
        """Extract HOG features for all images"""
        print("\nExtracting HOG features...")
        features = []
        
        for img_flat in tqdm(X, desc="Extracting HOG"):
            # Reshape de (2304,) à (48, 48)
            img = img_flat.reshape(48, 48)
            feat = self.extract_hog_features(img)
            features.append(feat)
        
        return np.array(features)
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train Linear SVM on HOG features"""
        print("\nExtracting HOG features for training set...")
        X_train_hog = self.extract_features_batch(X_train)
        
        print(f"Feature shape: {X_train_hog.shape}")
        
        # Train Linear SVM with One-vs-Rest strategy
        print("\nTraining Linear SVM (OvR)...")
        self.model = LinearSVC(
            C=0.1,  # Regularization parameter
            max_iter=5000,
            dual=False,  # Prefer dual=False when n_samples > n_features
            random_state=42,
            verbose=1
        )
        
        self.model.fit(X_train_hog, y_train)
        
        # Evaluate on training set
        train_pred = self.model.predict(X_train_hog)
        train_acc = accuracy_score(y_train, train_pred)
        print(f"\nTraining accuracy: {train_acc:.4f}")
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            print("\nExtracting HOG features for validation set...")
            X_val_hog = self.extract_features_batch(X_val)
            
            val_pred = self.model.predict(X_val_hog)
            val_acc = accuracy_score(y_val, val_pred)
            print(f"Validation accuracy: {val_acc:.4f}")
            
            print("\nValidation Classification Report:")
            print(classification_report(y_val, val_pred, target_names=[e.capitalize() for e in EMOTIONS]))
            
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_val, val_pred))
        
        return self.model
    
    def save_model(self, path='hog_svm_model.pkl'):
        """Save trained model to file"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"\nModel saved to {path}")
    
    def load_model(self, path='hog_svm_model.pkl'):
        """Load trained model from file"""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {path}")

def main():
    # Configuration
    DATASET_PATH = "archive"  # Chemin vers le dossier archive
    
    if not os.path.exists(DATASET_PATH):
        print("❌ Dataset folder 'archive' not found!")
        print("\nExpected structure:")
        print("archive/")
        print("  train/")
        print("    angry/")
        print("    disgust/")
        print("    fear/")
        print("    happy/")
        print("    sad/")
        print("    surprise/")
        print("    neutral/")
        print("  test/")
        print("    angry/")
        print("    ...")
        return
    
    # Initialize trainer
    trainer = HOGSVMTrainer()
    
    # Load dataset
    X_train, y_train, X_test, y_test = trainer.load_fer2013_from_folders(DATASET_PATH)
    
    if len(X_train) == 0:
        print("❌ No training data loaded!")
        return
    
    # Optional: Create validation set from training data
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    
    print(f"\nFinal split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Train model
    trainer.train(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    if len(X_test) > 0:
        print("\n" + "="*50)
        print("Evaluating on test set...")
        X_test_hog = trainer.extract_features_batch(X_test)
        test_pred = trainer.model.predict(X_test_hog)
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f"\n🎯 Test accuracy: {test_acc:.4f}")
        print("\nTest Classification Report:")
        print(classification_report(y_test, test_pred, target_names=[e.capitalize() for e in EMOTIONS]))
    
    # Save model
    trainer.save_model('hog_svm_model.pkl')
    
    print("\n✅ Training complete!")
    print(f"Expected accuracy: 30-45% (HOG+SVM is a baseline)")
    if len(X_test) > 0:
        print(f"Actual test accuracy: {test_acc:.2%}")

if __name__ == "__main__":
    main()