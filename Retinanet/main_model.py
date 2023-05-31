# Object Detection with RetinaNet

#**Author:**  TomProud & Yasser388 inspired by the work of Srihari Humbarwadi

#**Description:**  Implementing RetinaNet: Focal Loss for Dense Object Detection.
## Introduction

#Inspired from the Object Detection with RetinaNet our ambition is to test the model on a new dataset: Cityscapes. Improve the performance of the model by modifying the backbone and ultimately to modify the model architecture. 

#**References:**

#- [RetinaNet Paper](https://arxiv.org/abs/1708.02002)
#- [Feature Pyramid Network Paper](https://arxiv.org/abs/1612.03144)

import os
import glob
import json

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

from function.anchor_generator import *
from function.backbone_resnet50 import *
from function.class_box_heads import *
from function.iou import *
from function.L1_Loss_Focal_Loss import *
from function.network_architecture import *
from function.labels import *
from function.pred import *
from function.preprocessing_data import *
from function.retinanet_model import *
from function.utility_functions import *

##label

label_id_name_mapping = {
    
    1: 'none',
    2: 'person',
    3: 'rider',
    4: 'car',
    5: 'truck',
    6: 'bus',
    7: 'train',
    8: 'motorcycle',
    9: 'bicycle',       
   10: 'traffic light',
   11: 'traffic sign',
   12: 'caravan',
   13: 'license plate',
	 14: 'train',
}


## Setting up training parameters

model_dir = "retinanet/"
label_encoder = LabelEncoder()

num_classes = 15
batch_size = 1

learning_rates = [1e-05, 0.0025, 0.005, 0.01, 0.001, 0.0001]
learning_rate_boundaries = [100, 200, 400, 200000, 300000]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries, values=learning_rates
)

## Initializing and compiling model
resnet50_backbone = get_backbone()
loss_fn = RetinaNetLoss(num_classes)
model = RetinaNet(num_classes, resnet50_backbone)

optimizer = tf.optimizers.Adam(learning_rate=0.01)
model.compile(loss=loss_fn, optimizer=optimizer)
## Setting up callbacks
callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
        monitor="loss",
        save_best_only=False,
        save_weights_only=True,
        verbose=1,
    )
]

## Import Cityscapes & Merges Images and .Json file

#Dataset_train & Dataset_val 
# Read the JSON file
with open('conversion_seg_to_bbox/data/cityscapes/annotations/instancesonly_filtered_gtFine_train.json') as f:
    data = json.load(f)

images_dir = "conversion_seg_to_bbox/data/cityscapes/"

def load_and_preprocess_image(images_dir, file_name):
    # Construct the full path to the image file
    image_path = os.path.join(images_dir, file_name)
    
    # Read image from file
    image = tf.io.read_file(image_path)
    
    # Decode image to tensor
    image = tf.image.decode_image(image, channels=3)
    
    return image

# Create an empty list to store image data
image_data = []

# Iterate over the images in the JSON file
for image_info in data['images']:
    # Extract image information
    file_name = image_info['file_name']
    image_id = image_info['id']
    
    # Load and preprocess the image
    image = load_and_preprocess_image(images_dir, file_name)
    
    # Extract object information for the current image
    objects = [obj for obj in data['annotations'] if obj['image_id'] == image_id]
    
    # Skip images with no bounding boxes
    if len(objects) == 0:
        continue
    
    areas = np.array([obj['area'] for obj in objects])
    bboxes = np.array([obj['bbox'] for obj in objects])
    id = np.array([obj['id'] for obj in objects])
    iscrowd = np.array([obj['iscrowd'] for obj in objects])
    labels = np.array([obj['category_id'] for obj in objects])
    
    # Create a dictionary for the current image
    image_dict = {
        'image': image,
        'image/filename': file_name,
        'image/id': image_id,
        'objects': {
            'area': areas,
            'bbox': bboxes,
            'id': id,
            'iscrowd': iscrowd,
            'label': labels
        }
    }
    
    # Append the image dictionary to the list
    image_data.append(image_dict)

# Create a dataset from the image data
dataset_train = tf.data.Dataset.from_generator(
    lambda: image_data,
    output_signature={
        'image': tf.TensorSpec(shape=(None,None,3), dtype=tf.uint8),
        'image/filename': tf.TensorSpec(shape=(), dtype=tf.string),
        'image/id': tf.TensorSpec(shape=(), dtype=tf.int64),
        'objects': {
            'area': tf.TensorSpec(shape=(None,), dtype=tf.int64),
            'bbox': tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            'id': tf.TensorSpec(shape=(None,), dtype=tf.int64),
            'iscrowd': tf.TensorSpec(shape=(None,), dtype=tf.bool),
            'label': tf.TensorSpec(shape=(None,), dtype=tf.int64),
        }
    }
)

import json
import numpy as np
import os
import tensorflow as tf

# Read the JSON file
with open('conversion_seg_to_bbox/data/cityscapes/annotations/instancesonly_filtered_gtFine_val.json') as f:
    data = json.load(f)

images_dir = "conversion_seg_to_bbox/data/cityscapes/"

def load_and_preprocess_image(images_dir, file_name):
    # Construct the full path to the image file
    image_path = os.path.join(images_dir, file_name)
    
    # Read image from file
    image = tf.io.read_file(image_path)
    
    # Decode image to tensor
    image = tf.image.decode_image(image, channels=3)
    
    return image

# Create an empty list to store image data
image_data = []

# Iterate over the images in the JSON file
for image_info in data['images']:
    # Extract image information
    file_name = image_info['file_name']
    image_id = image_info['id']
    
    # Load and preprocess the image
    image = load_and_preprocess_image(images_dir, file_name)
    
    # Extract object information for the current image
    objects = [obj for obj in data['annotations'] if obj['image_id'] == image_id]
    
    # Skip images with no bounding boxes
    if len(objects) == 0:
        continue
    
    areas = np.array([obj['area'] for obj in objects])
    bboxes = np.array([obj['bbox'] for obj in objects])
    id = np.array([obj['id'] for obj in objects])
    iscrowd = np.array([obj['iscrowd'] for obj in objects])
    labels = np.array([obj['category_id'] for obj in objects])
    
    # Create a dictionary for the current image
    image_dict = {
        'image': image,
        'image/filename': file_name,
        'image/id': image_id,
        'objects': {
            'area': areas,
            'bbox': bboxes,
            'id': id,
            'iscrowd': iscrowd,
            'label': labels
        }
    }
    
    # Append the image dictionary to the list
    image_data.append(image_dict)

# Create a dataset from the image data
dataset_val = tf.data.Dataset.from_generator(
    lambda: image_data,
    output_signature={
        'image': tf.TensorSpec(shape=(None,None,3), dtype=tf.uint8),
        'image/filename': tf.TensorSpec(shape=(), dtype=tf.string),
        'image/id': tf.TensorSpec(shape=(), dtype=tf.int64),
        'objects': {
            'area': tf.TensorSpec(shape=(None,), dtype=tf.int64),
            'bbox': tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            'id': tf.TensorSpec(shape=(None,), dtype=tf.int64),
            'iscrowd': tf.TensorSpec(shape=(None,), dtype=tf.bool),
            'label': tf.TensorSpec(shape=(None,), dtype=tf.int64),
        }
    }
)


## Setting up a `tf.data` pipeline

#- Apply the preprocessing function to the samples
#- Create batches with fixed batch size. Since images in the batch can
#have different dimensions, and can also have different number of
#objects, we use `padded_batch` to the add the necessary padding to create
#rectangular tensors
#- Create targets for each sample in the batch using `LabelEncoder`

train_dataset = dataset_train
val_dataset = dataset_val
autotune = tf.data.AUTOTUNE
train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=autotune)
train_dataset = train_dataset.shuffle(8 * batch_size)
train_dataset = train_dataset.padded_batch(
    batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True
)
train_dataset = train_dataset.map(
    label_encoder.encode_batch, num_parallel_calls=autotune
)
train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
train_dataset = train_dataset.prefetch(autotune)

val_dataset = val_dataset.map(preprocess_data, num_parallel_calls=autotune)
val_dataset = val_dataset.padded_batch(
    batch_size=1, padding_values=(0.0, 1e-8, -1), drop_remainder=True
)
val_dataset = val_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
val_dataset = val_dataset.prefetch(autotune)

## Training the model

epochs = 5  # Total number of epochs

# Run the training loop
history = model.fit(
    train_dataset.take(100),
    validation_data=val_dataset.take(50),
    epochs=epochs,
    callbacks=callbacks_list,
    verbose=1,
)

print(history.history.keys())
print(history.history['val_loss'])
