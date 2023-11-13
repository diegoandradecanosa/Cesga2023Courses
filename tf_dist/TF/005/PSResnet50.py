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
#from tensorflow.distribute.cluster_resolver import SlurmClusterResolver
import json
import os


def run_ps(task_index, cluster):
  server = tf.distribute.Server(cluster.as_cluster_def(),
                           job_name='ps',
                           task_index=task_index,
                           protocol='grpc')
  server.join()

def run_worker(task_index, cluster):

  server = tf.distribute.Server(cluster.as_cluster_def(),
                           job_name='worker',
                           task_index=task_index,
                           protocol='grpc')
  server.join()

cluster_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver(
      {'ps': 1, 'worker': int(os.environ['SLURM_NTASKS'])-1},
      port_base=8888,
      tasks_per_node=2,
      gpus_per_node=2,
      gpus_per_task=1,
      auto_set_gpu=True)

cluster = cluster_resolver.cluster_spec()
job_name, task_index = cluster_resolver.get_task_info()

if job_name == 'ps':
    run_ps(task_index, cluster)
else:
    run_worker(task_index, cluster)
    
strategy = tf.distribute.ParameterServerStrategy(
    cluster_resolver=cluster_resolver)
  
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