# Import necessary libraries
import os  # For handling file and directory operations
import numpy as np  # For numerical operations and array handling
import matplotlib.pyplot as plt  # For plotting and displaying images
import matplotlib.image as mpimg  # For reading images
import cv2  # OpenCV library for image processing
from PIL import Image  # Python Imaging Library for image operations
from sklearn.model_selection import train_test_split  # For splitting dataset into training and testing sets

# Set up data paths using os.path.join for cross-platform compatibility
# This creates proper path strings regardless of operating system
with_mask_path = os.path.join('data', 'with_mask')
without_mask_path = os.path.join('data', 'without_mask')

# Get lists of all files in both directories
with_mask_files = os.listdir(with_mask_path)
without_mask_files = os.listdir(without_mask_path)

# Print dataset statistics
print('Number of with mask images:', len(with_mask_files))
print('Number of without mask images:', len(without_mask_files))

# Create labels for the images
# Label 1: represents images with masks
# Label 0: represents images without masks
with_mask_labels = [1] * len(with_mask_files)
without_mask_labels = [0] * len(without_mask_files)
labels = with_mask_labels + without_mask_labels  # Combine all labels into one list

# Initialize empty list to store processed images
data = []

# Process images with masks
for img_file in with_mask_files:
    # Open each image using PIL (Python Imaging Library)
    image = Image.open(os.path.join(with_mask_path, img_file))
    # Resize image to 128x128 pixels for consistent input size
    image = image.resize((128,128))
    # Convert image to RGB format (3 channels)
    image = image.convert('RGB')
    # Convert PIL image to numpy array for neural network processing
    image = np.array(image)
    data.append(image)

# Process images without masks (same steps as above)
for img_file in without_mask_files:
    image = Image.open(os.path.join(without_mask_path, img_file))
    image = image.resize((128,128))
    image = image.convert('RGB')
    image = np.array(image)
    data.append(image)

# Convert lists to numpy arrays for machine learning processing
X = np.array(data)  # Features (images)
Y = np.array(labels)  # Labels

# Split data into training and testing sets
# test_size=0.2 means 20% of data will be used for testing
# random_state=2 ensures reproducibility of the split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Scale pixel values to range [0,1] by dividing by 255 (max pixel value)
X_train_scaled = X_train/255
X_test_scaled = X_test/255

# Import TensorFlow and Keras for deep learning
import tensorflow as tf
from tensorflow import keras
num_of_classes = 2  # Binary classification (mask/no mask)

# Define the CNN model architecture using Sequential API
model = keras.Sequential([
    # First Convolutional Layer
    # 32 filters, 3x3 kernel size, ReLU activation, input shape matches our image dimensions
    keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(128,128,3)),
    # Max Pooling layer to reduce spatial dimensions
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    
    # Second Convolutional Layer
    # 64 filters, 3x3 kernel size, ReLU activation
    keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    
    # Flatten layer to convert 2D feature maps to 1D feature vector
    keras.layers.Flatten(),
    
    # Dense layers for classification
    # 128 neurons, ReLU activation
    keras.layers.Dense(128, activation='relu'),
    # Dropout layer to prevent overfitting (50% dropout rate)
    keras.layers.Dropout(0.5),
    
    # Second Dense layer
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    
    # Output layer with sigmoid activation for binary classification
    keras.layers.Dense(num_of_classes, activation='sigmoid')
])

# Compile the model
# optimizer='adam': Uses Adam optimization algorithm
# loss='sparse_categorical_crossentropy': Appropriate for integer labels
# metrics=['acc']: Track accuracy during training
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['acc'])

# Train the model
# validation_split=0.1: Uses 10% of training data for validation
# epochs=5: Number of complete passes through the training dataset
history = model.fit(X_train_scaled, Y_train, validation_split=0.1, epochs=5)

# Evaluate model on test data
loss, accuracy = model.evaluate(X_test_scaled, Y_test)
print('Test Accuracy =', accuracy)

# Plot training history
plt.figure(figsize=(12, 4))  # Set figure size

# Plot training and validation loss
plt.subplot(1, 2, 1)  # First subplot
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.legend()
plt.title('Training Loss')

# Plot training and validation accuracy
plt.subplot(1, 2, 2)  # Second subplot
plt.plot(history.history['acc'], label='train accuracy')
plt.plot(history.history['val_acc'], label='validation accuracy')
plt.legend()
plt.title('Training Accuracy')
plt.show()

# Function to predict mask presence in new images
def predict_mask(image_path):
    # Read image using OpenCV
    input_image = cv2.imread(image_path)
    # Convert BGR to RGB and display image
    plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    
    # Preprocess image for prediction
    input_image_resized = cv2.resize(input_image, (128,128))  # Resize to match training size
    input_image_scaled = input_image_resized/255  # Scale pixel values
    input_image_reshaped = np.reshape(input_image_scaled, [1,128,128,3])  # Add batch dimension
    
    # Make prediction
    input_prediction = model.predict(input_image_reshaped)
    input_pred_label = np.argmax(input_prediction)  # Get class with highest probability
    
    # Convert prediction to human-readable result
    result = 'The person in the image is not wearing a mask' if input_pred_label == 1 else 'The person in the image is wearing a mask'
    print(result)

# Interactive prediction loop
while True:
    try:
        # Get image path from user
        image_path = input('\nEnter path of the image to predict (or "q" to quit): ')
        if image_path.lower() == 'q':
            break
        predict_mask(image_path)
    except Exception as e:
        print(f"Error processing image: {e}")