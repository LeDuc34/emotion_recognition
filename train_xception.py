import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Emotion labels for FER-2013
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
NUM_CLASSES = len(EMOTIONS)

class MiniXceptionTrainer:
    """Trainer for Mini-Xception CNN on FER-2013"""
    def __init__(self, input_shape=(48, 48, 1)):
        self.input_shape = input_shape
        self.model = None
        self.history = None
    
    def build_mini_xception(self):
        """
        Build Mini-Xception architecture
        Inspired by the Xception paper but scaled down for 48x48 images
        """
        inputs = layers.Input(shape=self.input_shape)
        
        # Entry flow
        x = layers.Conv2D(8, (3, 3), strides=(1, 1), use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(8, (3, 3), strides=(1, 1), use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Module 1
        residual = layers.Conv2D(16, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = layers.BatchNormalization()(residual)
        
        x = layers.SeparableConv2D(16, (3, 3), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(16, (3, 3), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        
        x = layers.add([x, residual])
        
        # Module 2
        residual = layers.Conv2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = layers.BatchNormalization()(residual)
        
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(32, (3, 3), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(32, (3, 3), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        
        x = layers.add([x, residual])
        
        # Module 3
        residual = layers.Conv2D(64, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = layers.BatchNormalization()(residual)
        
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(64, (3, 3), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(64, (3, 3), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        
        x = layers.add([x, residual])
        
        # Module 4
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        
        # Exit flow
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def train(self, train_dir, test_dir, epochs=100, batch_size=64):
        """
        Train Mini-Xception model using ImageDataGenerator
        
        Args:
            train_dir: path to training folder (archive/train)
            test_dir: path to test folder (archive/test)
        """
        
        # Build model
        print("\nBuilding Mini-Xception model...")
        self.model = self.build_mini_xception()
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(self.model.summary())
        
        # Data augmentation pour l'entraînement
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=False,  # Les émotions ne sont pas symétriques
            validation_split=0.15  # 15% pour validation
        )
        
        # Pas d'augmentation pour le test
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Générateurs de données
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(48, 48),
            batch_size=batch_size,
            color_mode='grayscale',
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        validation_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(48, 48),
            batch_size=batch_size,
            color_mode='grayscale',
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(48, 48),
            batch_size=batch_size,
            color_mode='grayscale',
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"\nDataset info:")
        print(f"  Training samples: {train_generator.samples}")
        print(f"  Validation samples: {validation_generator.samples}")
        print(f"  Test samples: {test_generator.samples}")
        print(f"  Classes: {train_generator.class_indices}")
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                'mini_xception_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train
        print("\nTraining Mini-Xception...")
        self.history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        print("\n" + "="*50)
        print("Evaluating on test set...")
        test_loss, test_acc = self.model.evaluate(test_generator, verbose=0)
        print(f"\n🎯 Test accuracy: {test_acc:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        
        # Detailed predictions
        print("\nGenerating predictions for classification report...")
        test_generator.reset()
        y_pred = self.model.predict(test_generator, verbose=1)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = test_generator.classes
        
        print("\nTest Classification Report:")
        print(classification_report(y_true, y_pred_classes, 
                                   target_names=[e.capitalize() for e in EMOTIONS]))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred_classes))
        
        return self.model
    
    def save_model(self, path='mini_xception_model.h5'):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        self.model.save(path)
        print(f"\nModel saved to {path}")
    
    def load_model(self, path='mini_xception_model.h5'):
        """Load trained model"""
        self.model = keras.models.load_model(path)
        print(f"Model loaded from {path}")

def main():
    # Configuration
    DATASET_PATH = "archive"
    TRAIN_DIR = os.path.join(DATASET_PATH, "train")
    TEST_DIR = os.path.join(DATASET_PATH, "test")
    EPOCHS = 100
    BATCH_SIZE = 64
    
    # Vérifier que les dossiers existent
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ Training folder not found: {TRAIN_DIR}")
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
        return
    
    if not os.path.exists(TEST_DIR):
        print(f"❌ Test folder not found: {TEST_DIR}")
        return
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Initialize trainer
    trainer = MiniXceptionTrainer()
    
    # Train model
    trainer.train(TRAIN_DIR, TEST_DIR, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    # Save model
    trainer.save_model('mini_xception_model.h5')
    
    print("\n✅ Training complete!")
    print(f"Target accuracy: 60-67%")
    
    # Print training history summary
    if trainer.history:
        best_val_acc = max(trainer.history.history['val_accuracy'])
        print(f"\nBest validation accuracy during training: {best_val_acc:.2%}")

if __name__ == "__main__":
    main()