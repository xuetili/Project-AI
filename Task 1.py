# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 11:59:14 2025

@author: Marcin
"""


import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- 1. Loading data ---

#paths
train_dir = 'covid xray/train'
val_dir = 'covid xray/val'
test_dir = 'covid xray/test'

# image generators for training, validation, and test data
train_image_generator = ImageDataGenerator(rescale=1./255)
val_image_generator = ImageDataGenerator(rescale=1./255)
test_image_generator = ImageDataGenerator(rescale=1./255)

batch_size = 128

# Load as arrays (flow_from_directory method)
train_generator = train_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    shuffle=True,
    target_size=(299, 299),
    class_mode='binary'
)

val_generator = val_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=val_dir,
    shuffle=True,
    target_size=(299, 299),
    class_mode='binary'
)

test_generator = test_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=test_dir,
    shuffle=True,
    target_size=(299, 299),
    class_mode='binary'
)

# --- 2. Data exploration ---

# Get training dataset size
train_size = len(train_generator.filenames)
print(f"Training dataset size: {train_size}")

# Check the size of a sample image
sample_batch, labels = next(train_generator)
print(f"Sample image shape: {sample_batch.shape}")

# Distribution of classes in the training set
class_labels = train_generator.class_indices
class_counts = {label: sum(labels == i) for i, label in enumerate(class_labels)}
plt.bar(class_counts.keys(), class_counts.values())
plt.xlabel('Class')
plt.ylabel('Number of samples')
plt.title('Class Distribution in Training Set')
plt.show()


# Plot some sample images from each class
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i in range(5):
    axes[i].imshow(sample_batch[i])
    axes[i].set_title("COVID" if labels[i] == 1 else "Normal")
    axes[i].axis('off')
plt.show()

# Calculate pixel average and standard deviation for the training data
pixel_mean = np.mean(sample_batch)
pixel_std = np.std(sample_batch)

print(f"Pixel Average: {pixel_mean:.2f}")
print(f"Pixel Standard Deviation: {pixel_std:.2f}")

# Repeat for validation dataset
val_batch, val_labels = next(val_generator)
val_mean = np.mean(val_batch)
val_std = np.std(val_batch)

print(f"Validation Pixel Average: {val_mean:.2f}")
print(f"Validation Pixel Standard Deviation: {val_std:.2f}")

# Repeat for test dataset
test_batch, test_labels = next(test_generator)
test_mean = np.mean(test_batch)
test_std = np.std(test_batch)

print(f"Test Pixel Average: {test_mean:.2f}")
print(f"Test Pixel Standard Deviation: {test_std:.2f}")


## --- 3. PreProcessing ---

# rescale size
img_size = (128, 128)

# Use training dataset statistics for normalization
train_image_generator = ImageDataGenerator(rescale=1./255, 
                                           featurewise_center=True, 
                                           featurewise_std_normalization=True)
train_image_generator.mean = pixel_mean
train_image_generator.std = pixel_std  

# Validation and test sets use the same mean and std as the training set
val_image_generator = ImageDataGenerator(rescale=1./255, 
                                         featurewise_center=True, 
                                         featurewise_std_normalization=True)
val_image_generator.mean = pixel_mean 
val_image_generator.std = pixel_std    

test_image_generator = ImageDataGenerator(rescale=1./255, 
                                          featurewise_center=True, 
                                          featurewise_std_normalization=True)
test_image_generator.mean = pixel_mean 
test_image_generator.std = pixel_std  

# Load datasets with new preprocessing
train_generator = train_image_generator.flow_from_directory(
    train_dir, target_size=img_size, batch_size=128, class_mode='binary')
val_generator = val_image_generator.flow_from_directory(
    val_dir, target_size=img_size, batch_size=128, class_mode='binary')
test_generator = test_image_generator.flow_from_directory(
    test_dir, target_size=img_size, batch_size=128, class_mode='binary')

# --- Check Normalization on Validation Sample ---
val_sample, _ = next(val_generator)
print(f"Validation mean after normalization: {np.mean(val_sample):.2f}")
print(f"Validation std after normalization: {np.std(val_sample):.2f}")


# --- 4. Data Augmentation ---

# Function to generate augmented images
def get_augmented_images(datagen, train_dir, batch_size=32, img_size=(128, 128), num_images=5):
    train_data_gen = datagen.flow_from_directory(
        batch_size=batch_size,
        directory=train_dir,
        shuffle=True,
        target_size=img_size
    )
    augmented_images = [train_data_gen[0][0][0] for _ in range(num_images)]
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    for img, ax in zip(augmented_images , axes):
        ax.imshow(img)
        ax.axis('off')
    plt.show()
    return [train_data_gen[0][0][0] for _ in range(num_images)]  

# Horizontal Flip 
flip_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
flip_images = get_augmented_images(flip_gen, train_dir)

# Rotation 
rotate_gen = ImageDataGenerator(rescale=1./255, rotation_range=15)
rotate_images = get_augmented_images(rotate_gen, train_dir)

# Zoom 
zoom_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.2)
zoom_images = get_augmented_images(zoom_gen, train_dir)

# Combined 
combined_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True
)

combined_images = get_augmented_images(combined_gen, train_dir)


## also plot class labels for val and test (Question 2)
# --- Validation set class distribution ---
val_labels_batch = val_labels  # already loaded from earlier
val_class_counts = {label: sum(val_labels_batch == i) for i, label in enumerate(class_labels)}
plt.bar(val_class_counts.keys(), val_class_counts.values())
plt.xlabel('Class')
plt.ylabel('Number of samples')
plt.title('Class Distribution in Validation Set')
plt.show()

# --- Test set class distribution ---
test_labels_batch = test_labels  # already loaded from earlier
test_class_counts = {label: sum(test_labels_batch == i) for i, label in enumerate(class_labels)}
plt.bar(test_class_counts.keys(), test_class_counts.values())
plt.xlabel('Class')
plt.ylabel('Number of samples')
plt.title('Class Distribution in Test Set')
plt.show()

