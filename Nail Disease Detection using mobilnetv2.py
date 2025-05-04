#!/usr/bin/env python
# coding: utf-8

# In[13]:


#new model 


# In[14]:


import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set paths
train_dir = r"C:\Users\PC\Downloads\nail disease detection\data\train"
val_dir = r"C:\Users\PC\Downloads\nail disease detection\data\validation"

# Data augmentation & preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1./255)

image_size = (224, 224)
batch_size = 32

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Load MobileNetV2 base
base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model

# Custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)  # Increased Dropout to prevent overfitting
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)  # Increased Dropout
predictions = Dense(train_generator.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
initial_epochs = 15
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=initial_epochs,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)
    ]
)

# Fine-tuning the base model
base_model.trainable = True  # Unfreeze the base model layers
for layer in base_model.layers[:100]:  # Freeze first 100 layers (fine-tuning later layers)
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

fine_tune_epochs = 15
history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=fine_tune_epochs,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)
    ]
)

# Save the retrained model
model.save("nail_disease_retrained_mobilenetv2.h5")

# Evaluate the model
val_generator.reset()
preds = model.predict(val_generator, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

# Classification Report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[15]:


# Assuming the test set is separate from the validation set, and the path is defined
test_dir = r"C:\Users\PC\Downloads\nail disease detection\data\test"

# Data augmentation & preprocessing (just rescale for the test set)
test_datagen = ImageDataGenerator(rescale=1./255)

# Test set generator
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Important: Don't shuffle the test set
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)

# Print the test accuracy and loss
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Make predictions on the test set
test_preds = model.predict(test_generator, verbose=1)
y_pred_test = np.argmax(test_preds, axis=1)
y_true_test = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Classification Report on the test set
print("Classification Report (Test Set):")
print(classification_report(y_true_test, y_pred_test, target_names=class_labels))

# Confusion Matrix for the test set
cm_test = confusion_matrix(y_true_test, y_pred_test)

# Plot Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix (Test Set)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[3]:


# After training your model, save it as .h5 format
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('nail_disease_retrained_mobilenetv2.h5')

# Ensure model is trained before this step.
model.save(r'C:\Users\PC\OneDrive\Documents\nail_disease_detection\nail_disease_retrained_mobilenetv2.h5')

# Optionally, print confirmation
print("Model saved to .h5 format.")



# In[ ]:


import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

# Load models
efficientnet_model = load_model('nail_disease_efficientnetb0_model.h5')
mobilenetv2_model = load_model('nail_disease_retrained_mobilenetv2.h5')
densenet_model = load_model('densenet201_nail_disease.h5')

# Create a function to load and preprocess image
def load_and_preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

# Prediction function
def predict(model, img):
    processed_img = load_and_preprocess_image(img)
    prediction = model.predict(processed_img)
    return prediction

# Streamlit UI
st.title("Nail Disease Prediction")
st.write("Upload an image of a nail to predict the disease.")

# Image upload
uploaded_image = st.file_uploader("Choose an image...", type="jpg")

if uploaded_image is not None:
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    model_choice = st.selectbox("Select a model", ["EfficientNet", "MobileNetV2", "DenseNet"])

    if st.button('Predict'):
        #if model_choice == "EfficientNet":
         #   prediction = predict(efficientnet_model, img)
        if model_choice == "MobileNetV2":
            prediction = predict(mobilenetv2_model, img)
        elif model_choice == "DenseNet":
            prediction = predict(densenet_model, img)

        # Assuming you have class labels
        class_labels = ['Acral_Lentiginous_Melanoma', 'Healthy_Nail', 'Onychogryphosis', 'blue_finger', 'clubbing', 'pitting']
        
        predicted_class = class_labels[np.argmax(prediction)]
        st.write(f"Prediction: {predicted_class}")
        st.write(f"Confidence: {np.max(prediction) * 100:.2f}%")

