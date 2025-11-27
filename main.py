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
        if self.model is None:
            return "N/A", 0.0, 0
            
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
        if self.model is None:
            return "N/A", 0.0, 0
            
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
            'n_points': 24,
            'radius': 3,
            'method': 'uniform'
        }
    
    def extract_lbp_features(self, image):
        """Extract LBP features from grayscale image"""
        lbp = local_binary_pattern(
            image, 
            self.lbp_params['n_points'],
            self.lbp_params['radius'],
            method=self.lbp_params['method']
        )
        
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
        if self.model is None:
            return "N/A", 0.0, 0
        
        start_time = time.time()
        features = self.extract_lbp_features(image)
        
        prediction = self.model.predict([features])[0]
        distances = self.model.kneighbors([features], return_distance=True)[0][0]
        
        eps = 1e-10
        weights = 1.0 / (distances + eps)
        probabilities = weights / weights.sum()
        confidence = probabilities[0]
        
        latency = (time.time() - start_time) * 1000
        
        return EMOTIONS[prediction], confidence, latency

class EmotionRecognitionSystem:
    """Main system for emotion recognition"""
    def __init__(self):
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize classifiers
        self.hog_svm = HOGSVMClassifier()
        self.mini_xception = MiniXceptionClassifier()
        self.lbp_knn = LBPKNNClassifier()
        
        # Load models
        self.load_models()
    
    def load_models(self):
        """Load pretrained models if available"""
        if os.path.exists('hog_svm_model.pkl'):
            with open('hog_svm_model.pkl', 'rb') as f:
                self.hog_svm.model = pickle.load(f)
            print("✓ HOG+SVM model loaded")
        else:
            print("⚠ HOG+SVM model not found")
        
        try:
            import tensorflow as tf
            if os.path.exists('mini_xception_model.h5'):
                self.mini_xception.model = tf.keras.models.load_model('mini_xception_model.h5')
                print("✓ Mini-Xception model loaded")
            else:
                print("⚠ Mini-Xception model not found")
        except ImportError:
            print("⚠ TensorFlow not installed")
        
        if os.path.exists('lbp_knn_model.pkl'):
            with open('lbp_knn_model.pkl', 'rb') as f:
                self.lbp_knn.model = pickle.load(f)
            print("✓ LBP+KNN model loaded")
        else:
            print("⚠ LBP+KNN model not found")
    
    def detect_and_crop_face(self, image):
        """Detect face in image and return cropped face"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None, None
        
        (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48, 48))
        
        return face_resized, (x, y, w, h)
    
    def process_video_frame(self, image):
        """Process video frame with real-time annotations"""
        if image is None:
            return None
        
        # Detect and crop face
        face, bbox = self.detect_and_crop_face(image)
        
        # Create output image
        output_image = image.copy()
        
        if face is None or bbox is None:
            # No face detected
            cv2.putText(output_image, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return output_image
        
        x, y, w, h = bbox
        
        # Draw rectangle around face
        cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Get predictions from all models
        hog_emotion, hog_conf, _ = self.hog_svm.predict(face)
        cnn_emotion, cnn_conf, _ = self.mini_xception.predict(face)
        lbp_emotion, lbp_conf, _ = self.lbp_knn.predict(face)
        
        # Determine text position (above or below face)
        y_offset = y - 120 if y > 150 else y + h + 30
        
        # Background rectangle for text
        bg_width = max(len(hog_emotion), len(cnn_emotion), len(lbp_emotion)) * 10 + 100
        cv2.rectangle(output_image, (x, y_offset), (x+bg_width, y_offset+110), (0, 0, 0), -1)
        
        # HOG+SVM prediction
        cv2.putText(output_image, f"HOG: {hog_emotion}", (x+5, y_offset+25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
        cv2.putText(output_image, f"{hog_conf:.1%}", (x+5, y_offset+45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)
        
        # Mini-Xception prediction
        cv2.putText(output_image, f"CNN: {cnn_emotion}", (x+5, y_offset+65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
        cv2.putText(output_image, f"{cnn_conf:.1%}", (x+5, y_offset+85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)
        
        # LBP+KNN prediction
        cv2.putText(output_image, f"LBP: {lbp_emotion}", (x+5, y_offset+105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 200), 1)
        cv2.putText(output_image, f"{lbp_conf:.1%}", (x+bg_width-80, y_offset+105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 200), 1)
        
        return output_image
    
    def process_image(self, image):
        """Process single image and return detailed predictions"""
        face, bbox = self.detect_and_crop_face(image)
        
        if face is None:
            return None, "No face detected in image", None, None, None
        
        image_with_box = image.copy()
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(image_with_box, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        hog_emotion, hog_conf, hog_latency = self.hog_svm.predict(face)
        cnn_emotion, cnn_conf, cnn_latency = self.mini_xception.predict(face)
        lbp_emotion, lbp_conf, lbp_latency = self.lbp_knn.predict(face)
        
        hog_result = f"**Emotion:** {hog_emotion}\n**Confidence:** {hog_conf:.2%}\n**Latency:** {hog_latency:.2f} ms"
        cnn_result = f"**Emotion:** {cnn_emotion}\n**Confidence:** {cnn_conf:.2%}\n**Latency:** {cnn_latency:.2f} ms"
        lbp_result = f"**Emotion:** {lbp_emotion}\n**Confidence:** {lbp_conf:.2%}\n**Latency:** {lbp_latency:.2f} ms"
        
        return image_with_box, hog_result, cnn_result, lbp_result, face

# Initialize system
system = EmotionRecognitionSystem()

def predict_emotion_static(image):
    """Process static image"""
    if image is None:
        return None, "Please provide an image", None, None, None
    return system.process_image(image)

def predict_emotion_video(image):
    """Process video stream frame"""
    if image is None:
        return None
    return system.process_video_frame(image)

# Create Gradio interface
with gr.Blocks(title="Emotion Recognition Comparison") as demo:
    gr.Markdown("""
    # 🎭 Emotion Recognition Model Comparison
    
    Compare three emotion recognition approaches:
    - **HOG + Linear SVM**: Traditional computer vision
    - **Mini-Xception (CNN)**: Deep learning
    - **LBP + KNN**: Texture-based classification
    """)
    
    with gr.Tabs():
        # Tab 1: Real-time Video Stream
        with gr.Tab("📹 Real-Time Webcam"):
            gr.Markdown("""
            ### Live Emotion Detection
            Real-time emotion recognition from your webcam with continuous predictions.
            """)
            
            with gr.Row():
                video_input = gr.Image(sources=["webcam"], 
                                      type="numpy", 
                                      label="Webcam Feed",
                                      streaming=True)
                video_output = gr.Image(label="Predictions (Live)", 
                                       type="numpy",
                                       streaming=True)
            
            gr.Markdown("""
            **🔴 Live Mode:** Predictions are displayed directly on the video stream.
            - **Blue text**: HOG+SVM prediction
            - **Orange text**: Mini-Xception prediction
            - **Pink text**: LBP+KNN prediction
            """)
            
            # Connect video stream
            video_input.stream(
                fn=predict_emotion_video,
                inputs=video_input,
                outputs=video_output,
                time_limit=60,
                stream_every=0.1
            )
        
        # Tab 2: Static Image Analysis
        with gr.Tab("📸 Static Image"):
            gr.Markdown("""
            ### Detailed Image Analysis
            Upload an image or capture a photo for detailed emotion analysis.
            """)
            
            with gr.Row():
                with gr.Column():
                    static_input = gr.Image(sources=["upload", "webcam"], 
                                          type="numpy", 
                                          label="Input Image",
                                          streaming=False)
                    analyze_btn = gr.Button("🔍 Analyze Emotions", variant="primary")
                
                with gr.Column():
                    static_output = gr.Image(label="Detected Face", type="numpy")
                    cropped_face = gr.Image(label="Cropped Face (48×48)", type="numpy")
            
            gr.Markdown("### 📊 Detailed Predictions")
            
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
            
            # Connect static analysis
            analyze_btn.click(
                fn=predict_emotion_static,
                inputs=static_input,
                outputs=[static_output, hog_output, cnn_output, lbp_output, cropped_face]
            )
    
    gr.Markdown("""
    ---
    ### 📝 Notes
    - Models trained on FER-2013 dataset (7 emotions)
    - Face detection: OpenCV Haar Cascade
    - All predictions on 48×48 grayscale images
    - Real-time mode updates ~10 times per second
    """)

if __name__ == "__main__":
    demo.launch(share=True)