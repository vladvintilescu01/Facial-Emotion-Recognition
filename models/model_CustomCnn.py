# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Force use of first NVIDIA GPU

import tensorflow as tf
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, PReLU
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import mixed_precision
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Verify GPU is detected
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPU Devices: ", tf.config.list_physical_devices('GPU'))

gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    raise RuntimeError("No GPU found. This script is GPU-only.")
else:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(len(gpus), "Physical GPUs,", len(tf.config.list_logical_devices('GPU')), "Logical GPUs")

# Enable mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(f'Compute dtype: {policy.compute_dtype}, Variable dtype: {policy.variable_dtype}')

seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

IMG_HEIGHT, IMG_WIDTH, CHANNELS = 160, 160, 3
BATCH_SIZE = 16  
EPOCHS = 35
NUM_CLASSES = 5 

FER_PATH = "../fer2013.csv"
data = pd.read_csv(FER_PATH)

train_data = data[data['Usage']=='Training']
val_data   = data[data['Usage']=='PublicTest']
test_data  = data[data['Usage']=='PrivateTest']

print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}, Test samples: {len(test_data)}")

def preprocess(df):
    imgs, labels = [], []
    for _, row in df.iterrows():
        img = np.array(row['pixels'].split(), dtype='float32').reshape(48,48)
        img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img /= 255.0
        imgs.append(img)
        labels.append(row['emotion'])
    return np.array(imgs), to_categorical(np.array(labels), num_classes=NUM_CLASSES)

X_train, y_train = preprocess(train_data)
X_val, y_val = preprocess(val_data)
X_test, y_test = preprocess(test_data)
print("Preprocessing complete!")

# Compute class weights
y_train_labels = np.argmax(y_train, axis=1)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
) 

train_gen = ImageDataGenerator(
    horizontal_flip=True,
).flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)

val_gen = ImageDataGenerator().flow(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)

def custom_cnn(input_shape=(160,160,3), num_classes=NUM_CLASSES, lr=1e-3):

    inputs = tf.keras.Input(shape=input_shape)
    
    # Block 1: Basic Edge Detection 
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization(dtype='float32')(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(dtype='float32')(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.10)(x)
    
    # Block 2: Simple Shapes & Textures
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(dtype='float32')(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(dtype='float32')(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    
    # Block 3: Facial Features
    x = tf.keras.layers.Conv2D(128, (5, 5), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(dtype='float32')(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(dtype='float32')(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Block 4: Full Emotion Patterns 
    x = tf.keras.layers.Conv2D(256, (5, 5), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(dtype='float32')(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(dtype='float32')(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Block 5: Advanced Emotion Patterns, like more faces in the same frame
    x = tf.keras.layers.Conv2D(512, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(dtype='float32')(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(dtype='float32')(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.45)(x)
    
    # Global Average Pooling
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Dense Layers
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.BatchNormalization(dtype='float32')(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.BatchNormalization(dtype='float32')(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='Custom_FER_CNN')
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=SGD(learning_rate=lr, momentum=0.9, nesterov=True),
        metrics=['accuracy']
    )
    
    return model

model = custom_cnn()
model.summary()
print(f"Total trainable parameters: {model.count_params():,}")

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stop, lr_scheduler],
    class_weight=class_weight_dict,
    verbose=1
)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

y_pred = model.predict(val_gen, verbose=1)
y_true = np.argmax(y_val, axis=1)
y_pred_classes = np.argmax(y_pred, axis=1)

emotion_labels = ['Angry','Happy','Sad','Surprise','Neutral']

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=emotion_labels))

cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(10,8))
sns.heatmap(
    cm.astype("float") / cm.sum(axis=1)[:, np.newaxis],
    annot=True, fmt=".2f", cmap="Blues",
    xticklabels=emotion_labels, yticklabels=emotion_labels
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Normalized)")
plt.tight_layout()
plt.show()

test_gen = ImageDataGenerator().flow(X_test, y_test, batch_size=BATCH_SIZE, shuffle=False)
test_loss, test_acc = model.evaluate(test_gen, verbose=1)

print(f"\nTest Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

model.save("CustomCNN_FER2013.h5")
print("Model saved successfully as CustomCNN_FER2013.h5")