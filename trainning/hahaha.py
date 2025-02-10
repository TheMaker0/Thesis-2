import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Set up parameters
img_height = 128  # Reduced image height for faster training
img_width = 128   # Reduced image width for faster training
batch_size = 64   # Increased batch size for faster training
epochs = 50       # Number of training epochs

# Define the paths to the dataset
train_dir = r'C:\Users\Zero\Desktop\dataset sugarcane\train'  # Path to training images
val_dir = r'C:\Users\Zero\Desktop\dataset sugarcane\val'  # Path to validation images

# Create ImageDataGenerators for loading and augmenting images
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    shear_range=0.3,
    zoom_range=0.3,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'  # To handle image edges after shifting
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'  # Assuming multi-class classification
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Build the CNN model with improved architecture
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(512, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    
    # Adding Dropout to prevent overfitting
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    
    # Output layer
    layers.Dense(train_generator.num_classes, activation='softmax')
])

# Compile the model with an improved optimizer and loss function
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate for better convergence
    loss='categorical_crossentropy',  # For multi-class classification
    metrics=['accuracy']
)

# Set up model checkpoint and early stopping
checkpoint = ModelCheckpoint(
    'cnn.h5', 
    monitor='val_accuracy', 
    verbose=1, 
    save_best_only=True, 
    mode='max'
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,  # Stop if validation loss doesn't improve after 5 epochs
    restore_best_weights=True
)

# Train the model with the added callbacks
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[checkpoint, early_stopping]  # Added early stopping to prevent overfitting
)

# Save the final model (optional, if needed)
model.save('cnn_final.h5')

print("Model training is complete, and the model is saved as cnn.h5")

# Plotting training and validation accuracy/loss
# Plot accuracy
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
