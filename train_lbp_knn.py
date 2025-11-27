import numpy as np
import cv2
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from skimage.feature import local_binary_pattern
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

class LBPKNNTrainer:
    """Trainer for LBP + KNN classifier"""
    def __init__(self):
        self.lbp_params = {
            'n_points': 24,      # Number of circularly symmetric neighbor points
            'radius': 3,         # Radius of circle
            'method': 'uniform'  # LBP method (uniform patterns only)
        }
        self.model = None
    
    def extract_lbp_features(self, image):
        """
        Extract LBP features from grayscale image
        
        LBP (Local Binary Patterns) captures texture by comparing
        each pixel with its neighbors in a circular pattern.
        """
        # Si l'image n'est pas déjà en 48x48, la redimensionner
        if image.shape != (48, 48):
            image = cv2.resize(image, (48, 48))
        
        # Compute LBP
        lbp = local_binary_pattern(
            image, 
            self.lbp_params['n_points'],
            self.lbp_params['radius'],
            method=self.lbp_params['method']
        )
        
        # Compute histogram of LBP patterns
        # For 'uniform' method with P points: (P + 2) bins
        n_bins = self.lbp_params['n_points'] + 2
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=n_bins,
            range=(0, n_bins),
            density=True  # Normalize histogram
        )
        
        return hist
    
    def load_fer2013_from_folders(self, base_path='archive'):
        """
        Load FER-2013 dataset from folder structure:
        archive/
          train/
            angry/, disgust/, fear/, happy/, sad/, surprise/, neutral/
          test/
            angry/, disgust/, fear/, happy/, sad/, surprise/, neutral/
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
        """Extract LBP features for all images"""
        print("\nExtracting LBP features...")
        features = []
        
        for img_flat in tqdm(X, desc="Extracting LBP"):
            # Reshape de (2304,) à (48, 48)
            img = img_flat.reshape(48, 48)
            feat = self.extract_lbp_features(img)
            features.append(feat)
        
        return np.array(features)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, use_grid_search=False):
        """
        Train KNN classifier on LBP features
        
        Args:
            X_train: Training images (flattened)
            y_train: Training labels
            X_val: Validation images (optional)
            y_val: Validation labels (optional)
            use_grid_search: Whether to use grid search for hyperparameter tuning
        """
        print("\nExtracting LBP features for training set...")
        X_train_lbp = self.extract_features_batch(X_train)
        
        print(f"Feature shape: {X_train_lbp.shape}")
        print(f"  (n_samples: {X_train_lbp.shape[0]}, n_features: {X_train_lbp.shape[1]})")
        
        # Train KNN
        print("\nTraining K-Nearest Neighbors...")
        
        if use_grid_search:
            print("Using Grid Search for hyperparameter tuning...")
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
            
            knn = KNeighborsClassifier()
            grid_search = GridSearchCV(
                knn, 
                param_grid, 
                cv=3, 
                scoring='accuracy',
                verbose=2,
                n_jobs=-1
            )
            
            grid_search.fit(X_train_lbp, y_train)
            self.model = grid_search.best_estimator_
            
            print(f"\nBest parameters found:")
            for param, value in grid_search.best_params_.items():
                print(f"  {param}: {value}")
        else:
            # Use default parameters (faster)
            self.model = KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',  # Weight by inverse distance
                metric='euclidean',
                n_jobs=-1  # Use all CPU cores
            )
            
            self.model.fit(X_train_lbp, y_train)
        
        # Evaluate on training set
        train_pred = self.model.predict(X_train_lbp)
        train_acc = accuracy_score(y_train, train_pred)
        print(f"\nTraining accuracy: {train_acc:.4f}")
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            print("\nExtracting LBP features for validation set...")
            X_val_lbp = self.extract_features_batch(X_val)
            
            val_pred = self.model.predict(X_val_lbp)
            val_acc = accuracy_score(y_val, val_pred)
            print(f"Validation accuracy: {val_acc:.4f}")
            
            print("\nValidation Classification Report:")
            print(classification_report(y_val, val_pred, target_names=[e.capitalize() for e in EMOTIONS]))
            
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_val, val_pred))
        
        return self.model
    
    def save_model(self, path='lbp_knn_model.pkl'):
        """Save trained model to file"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"\nModel saved to {path}")
    
    def load_model(self, path='lbp_knn_model.pkl'):
        """Load trained model from file"""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {path}")

def main():
    # Configuration
    DATASET_PATH = "archive"  # Chemin vers le dossier archive
    USE_GRID_SEARCH = False   # True = plus lent mais meilleurs hyperparamètres
    
    if not os.path.exists(DATASET_PATH):
        print("❌ Dataset folder 'archive' not found!")
        print("\nExpected structure:")
        print("archive/")
        print("  train/")
        print("    angry/, disgust/, fear/, happy/, sad/, surprise/, neutral/")
        print("  test/")
        print("    angry/, disgust/, fear/, happy/, sad/, surprise/, neutral/")
        return
    
    # Initialize trainer
    trainer = LBPKNNTrainer()
    
    # Load dataset
    X_train, y_train, X_test, y_test = trainer.load_fer2013_from_folders(DATASET_PATH)
    
    if len(X_train) == 0:
        print("❌ No training data loaded!")
        return
    
    # Optional: Create validation set from training data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    
    print(f"\nFinal split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Train model
    trainer.train(X_train, y_train, X_val, y_val, use_grid_search=USE_GRID_SEARCH)
    
    # Evaluate on test set
    if len(X_test) > 0:
        print("\n" + "="*50)
        print("Evaluating on test set...")
        X_test_lbp = trainer.extract_features_batch(X_test)
        test_pred = trainer.model.predict(X_test_lbp)
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f"\n🎯 Test accuracy: {test_acc:.4f}")
        print("\nTest Classification Report:")
        print(classification_report(y_test, test_pred, target_names=[e.capitalize() for e in EMOTIONS]))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, test_pred))
    
    # Save model
    trainer.save_model('lbp_knn_model.pkl')
    
    print("\n✅ Training complete!")
    print(f"Expected accuracy: 40-55% (LBP+KNN is a texture-based approach)")
    if len(X_test) > 0:
        print(f"Actual test accuracy: {test_acc:.2%}")
    
    print("\n💡 Tips:")
    print("  - LBP captures texture patterns in facial regions")
    print("  - KNN is fast at inference (no training phase)")
    print("  - Try USE_GRID_SEARCH=True for better hyperparameters (slower)")

if __name__ == "__main__":
    main()