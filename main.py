import gradio as gr
import cv2
import numpy as np
import time
from skimage.feature import hog, local_binary_pattern
from skimage import exposure
import pickle
import os

# Emotion labels for FER-2013
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

class HOGSVMClassifier:
    """HOG + Linear SVM classifier for emotion recognition"""
    def __init__(self):
        self.model = None
        self.hog_params = {
            'orientations': 9,
            'pixels_per_cell': (8, 8),
            'cells_per_block': (2, 2),
            'block_norm': 'L2-Hys'
        }
    
    def extract_hog_features(self, image):
        """Extract HOG features from grayscale image"""
        features = hog(image, 
                      orientations=self.hog_params['orientations'],
                      pixels_per_cell=self.hog_params['pixels_per_cell'],
                      cells_per_block=self.hog_params['cells_per_block'],
                      block_norm=self.hog_params['block_norm'],
                      visualize=False)
        return features
    
    def predict(self, image):
        """Predict emotion from image"""
        start_time = time.time()
        features = self.extract_hog_features(image)
        
        # Get prediction and confidence
        prediction = self.model.predict([features])[0]
        decision_values = self.model.decision_function([features])[0]
        
        # Convert decision values to probabilities using softmax
        exp_values = np.exp(decision_values - np.max(decision_values))
        probabilities = exp_values / exp_values.sum()
        
        confidence = probabilities[prediction]
        latency = (time.time() - start_time) * 1000  # Convert to ms
        
        return EMOTIONS[prediction], confidence, latency

class MiniXceptionClassifier:
    """Mini-Xception CNN classifier for emotion recognition"""
    def __init__(self):
        self.model = None
    
    def predict(self, image):
        """Predict emotion from image"""
        start_time = time.time()
        
        # Normalize image
        img = image.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1)
        
        # Get prediction
        predictions = self.model.predict(img, verbose=0)[0]
        prediction = np.argmax(predictions)
        confidence = predictions[prediction]
        
        latency = (time.time() - start_time) * 1000  # Convert to ms
        
        return EMOTIONS[prediction], confidence, latency

class LBPKNNClassifier:
    """LBP (Local Binary Patterns) + KNN classifier for emotion recognition"""
    def __init__(self):
        self.model = None
        self.lbp_params = {
            'n_points': 24,  # Number of circularly symmetric neighbor points
            'radius': 3,     # Radius of circle
            'method': 'uniform'  # LBP method
        }
    
    def extract_lbp_features(self, image):
        """Extract LBP features from grayscale image"""
        # Compute LBP
        lbp = local_binary_pattern(
            image, 
            self.lbp_params['n_points'],
            self.lbp_params['radius'],
            method=self.lbp_params['method']
        )
        
        # Compute histogram of LBP
        n_bins = self.lbp_params['n_points'] + 2
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=n_bins,
            range=(0, n_bins),
            density=True
        )
        
        return hist
    
    def predict(self, image):
        """Predict emotion from image"""
        
        start_time = time.time()
        features = self.extract_lbp_features(image)
        
        # Get prediction and probabilities
        prediction = self.model.predict([features])[0]
        
        # Get probabilities (distances converted to probabilities)
        distances = self.model.kneighbors([features], return_distance=True)[0][0]
        
        # Convert distances to probabilities (inverse distance weighting)
        # Smaller distance = higher probability
        eps = 1e-10  # To avoid division by zero
        weights = 1.0 / (distances + eps)
        probabilities = weights / weights.sum()
        confidence = probabilities[0]  # Confidence from nearest neighbor
        
        latency = (time.time() - start_time) * 1000  # Convert to ms
        
        return EMOTIONS[prediction], confidence, latency

class EmotionRecognitionSystem:
    """Main system for emotion recognition"""
    def __init__(self):
        # Initialize face detector (Haar Cascade)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize classifiers
        self.hog_svm = HOGSVMClassifier()
        self.mini_xception = MiniXceptionClassifier()
        self.lbp_knn = LBPKNNClassifier()
        
        # Try to load models
        self.load_models()
    
    def load_models(self):
        """Load pretrained models if available"""
        # Try to load HOG+SVM model
        if os.path.exists('hog_svm_model.pkl'):
            with open('hog_svm_model.pkl', 'rb') as f:
                self.hog_svm.model = pickle.load(f)
            print("✓ HOG+SVM model loaded")
        else:
            print("⚠ HOG+SVM model not found")
            print("  Train a model or provide 'hog_svm_model.pkl'")
        
        # Try to load Mini-Xception model
        try:
            import tensorflow as tf
            if os.path.exists('mini_xception_model.h5'):
                self.mini_xception.model = tf.keras.models.load_model('mini_xception_model.h5')
                print("✓ Mini-Xception model loaded")
            else:
                print("⚠ Mini-Xception model not found")
                print("  Train a model or provide 'mini_xception_model.h5'")
        except ImportError:
            print("⚠ TensorFlow not installed")
        
        # Try to load LBP+KNN model
        if os.path.exists('lbp_knn_model.pkl'):
            with open('lbp_knn_model.pkl', 'rb') as f:
                self.lbp_knn.model = pickle.load(f)
            print("✓ LBP+KNN model loaded")
        else:
            print("⚠ LBP+KNN model not found")
            print("  Train a model or provide 'lbp_knn_model.pkl'")
    
    def detect_and_crop_face(self, image):
        """Detect face in image and return cropped face"""
        # Convert to grayscale for face detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None, None
        
        # Get the largest face
        (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
        
        # Crop and resize face
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48, 48))
        
        return face_resized, (x, y, w, h)
    
    def process_image(self, image):
        """Process image and return predictions from all models"""
        # Detect and crop face
        face, bbox = self.detect_and_crop_face(image)
        
        if face is None:
            return None, "No face detected in image", None, None, None
        
        # Draw rectangle on original image
        image_with_box = image.copy()
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(image_with_box, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Get predictions from all models
        hog_emotion, hog_conf, hog_latency = self.hog_svm.predict(face)
        cnn_emotion, cnn_conf, cnn_latency = self.mini_xception.predict(face)
        lbp_emotion, lbp_conf, lbp_latency = self.lbp_knn.predict(face)
        
        # Format results
        hog_result = f"**Emotion:** {hog_emotion}\n**Confidence:** {hog_conf:.2%}\n**Latency:** {hog_latency:.2f} ms"
        cnn_result = f"**Emotion:** {cnn_emotion}\n**Confidence:** {cnn_conf:.2%}\n**Latency:** {cnn_latency:.2f} ms"
        lbp_result = f"**Emotion:** {lbp_emotion}\n**Confidence:** {lbp_conf:.2%}\n**Latency:** {lbp_latency:.2f} ms"
        
        return image_with_box, hog_result, cnn_result, lbp_result, face

# Initialize system
system = EmotionRecognitionSystem()

def predict_emotion(image):
    """Main function for Gradio interface"""
    if image is None:
        return None, "Please provide an image", None, None, None
    
    return system.process_image(image)

# Create Gradio interface
with gr.Blocks(title="Emotion Recognition Comparison") as demo:
    gr.Markdown("""
    # 🎭 Emotion Recognition Model Comparison
    
    This interface compares three approaches for facial emotion recognition:
    1. **HOG + Linear SVM**: Traditional computer vision approach
    2. **Mini-Xception (CNN)**: Deep learning approach trained on FER-2013
    3. **LBP + KNN**: Local Binary Patterns with K-Nearest Neighbors
    
    Upload an image or use your webcam to see predictions side-by-side with confidence scores and latency.
    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(sources=["upload", "webcam"], 
                                  type="numpy", 
                                  label="Input Image",
                                  streaming=False)
            predict_btn = gr.Button("🔍 Analyze Emotions", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(label="Detected Face", type="numpy")
            cropped_face = gr.Image(label="Cropped Face (48×48)", type="numpy")
    
    gr.Markdown("### 📊 Model Predictions")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("#### 🔷 HOG + Linear SVM")
            hog_output = gr.Markdown()
        
        with gr.Column():
            gr.Markdown("#### 🔶 Mini-Xception (CNN)")
            cnn_output = gr.Markdown()
        
        with gr.Column():
            gr.Markdown("#### 🔸 LBP + KNN")
            lbp_output = gr.Markdown()
    
    gr.Markdown("""
    ---
    ### 📝 Notes:
    - Models should be trained on FER-2013 dataset
    - Face detection uses OpenCV Haar Cascade
    - All predictions are made on 48×48 grayscale images
    - Latency includes preprocessing and inference time
    """)
    
    # Connect interface
    predict_btn.click(
        fn=predict_emotion,
        inputs=input_image,
        outputs=[output_image, hog_output, cnn_output, lbp_output, cropped_face]
    )
    
    # Add examples
    gr.Markdown("### 💡 Try with example images:")
    gr.Examples(
        examples=[
            # You can add example image paths here
        ],
        inputs=input_image
    )

if __name__ == "__main__":
    demo.launch(share=True)