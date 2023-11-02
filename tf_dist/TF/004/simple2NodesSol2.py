#!/usr/bin/env python
# coding: utf-8
# Based on: https://deepsense.ai/tensorflow-on-slurm-clusters/

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pathlib
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.distribute.cluster_resolver import SlurmClusterResolver
import json
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

def set_tf_config(resolver, environment=None):
    """Set the TF_CONFIG env variable from the given cluster resolver"""
    cfg = {
        'cluster': resolver.cluster_spec().as_dict(),
        'task': {
            'type': resolver.get_task_info()[0],
            'index': resolver.get_task_info()[1],
        },
        'rpc_layer': resolver.rpc_layer,
    }
    if environment:
        cfg['environment'] = environment
    os.environ['TF_CONFIG'] = json.dumps(cfg)

resolver = SlurmClusterResolver()
set_tf_config(resolver)
print(resolver)
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(cluster_resolver=resolver)

#cluster_spec=tf.distribute.cluster_resolver.SlurmClusterResolver().cluster_spec()
#print(cluster_spec)
#strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=cluster_spec)


dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)
img_height,img_width=180,180
batch_size=128

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


import matplotlib.pyplot as plt
class_names = train_ds.class_names

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(6):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")




with strategy.scope():
  resnet_model = Sequential()
  pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                   input_shape=(180,180,3),
                   pooling='avg',classes=5,
                   weights='imagenet')
  for layer in pretrained_model.layers:
        layer.trainable=False

  resnet_model.add(pretrained_model)
  resnet_model.add(Flatten())
  resnet_model.add(Dense(512, activation='relu'))
  resnet_model.add(Dense(5, activation='softmax'))
  resnet_model.summary()

resnet_model.compile(optimizer=Adam(learning_rate=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#history = resnet_model.fit(train_ds, validation_data=val_ds, epochs=10)
tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = './logs',histogram_freq = 1,profile_batch = '10,20')
history = resnet_model.fit(train_ds, validation_data=val_ds, epochs=10,callbacks = [tboard_callback])

fig1 = plt.gcf()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.axis(ymin=0.4,ymax=1)
plt.grid()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()



