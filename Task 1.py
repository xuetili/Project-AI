# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 11:59:14 2025

@author: Marcin
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#paths
train_dir = 'covid xray/train'
val_dir = 'covid xray/val'
test_dir = 'covid xray/test'

# image generators for training, validation, and test data
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Load datasets as arrays (flow_from_directory method)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='binary')
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(224, 224), batch_size=32, class_mode='binary')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode='binary')


# Get training dataset size
train_size = len(train_generator.filenames)
print(f"Training dataset size: {train_size}")

# Check the size of a sample image
sample_batch, labels = next(train_generator)
print(f"Sample image shape: {sample_batch.shape}")

# Get class labels from the generator
class_labels = train_generator.class_indices
print(f"Class labels: {class_labels}")

# Plot the distribution of classes in the training set
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

print(f"Pixel Average: {pixel_mean}")
print(f"Pixel Standard Deviation: {pixel_std}")

# Repeat for validation dataset
val_batch, val_labels = next(val_generator)
val_mean = np.mean(val_batch)
val_std = np.std(val_batch)

print(f"Validation Pixel Average: {val_mean}")
print(f"Validation Pixel Standard Deviation: {val_std}")

# Repeat for test dataset
test_batch, test_labels = next(test_generator)
test_mean = np.mean(test_batch)
test_std = np.std(test_batch)

print(f"Test Pixel Average: {test_mean}")
print(f"Test Pixel Standard Deviation: {test_std}")
