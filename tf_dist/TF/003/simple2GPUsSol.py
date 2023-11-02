#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('module load tensorflow')
get_ipython().system('hostname')
#Based on: https://chroniclesofai.com/transfer-learning-with-keras-resnet-50/
#and https://github.com/nachi-hebbar/Transfer-Learning-ResNet-Keras/blob/main/ResNet_50.ipynb
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pathlib
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import os
get_ipython().system('pip install -U tensorboard_plugin_profile')
# In[2]:

import os
num_threads = 64
os.environ["OMP_NUM_THREADS"] = "64"
os.environ["TF_NUM_INTRAOP_THREADS"] = "64"
os.environ["TF_NUM_INTEROP_THREADS"] = "64"

tf.config.threading.set_inter_op_parallelism_threads(
    num_threads
)
tf.config.threading.set_intra_op_parallelism_threads(
    num_threads
)
tf.config.set_soft_device_placement(True)

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
lustrepath=os.environ["LUSTRE"]
data_dir = tf.keras.utils.get_file('flower_photos', dataset_url,cache_dir=lustrepath, untar=True)
data_dir = pathlib.Path(data_dir)
img_height,img_width=180,180
batch_size=8


print("In[3]:")
print("data_dir: "+str(data_dir))
strategy=tf.distribute.MirroredStrategy()




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
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# In[6]:


physical_devices = tf.config.list_physical_devices('GPU') 
print(physical_devices)
#strategy=tf.distribute.MirroredStrategy()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

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

tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = './logs',histogram_freq = 1,profile_batch = '200,220')

history = resnet_model.fit(train_ds, validation_data=val_ds, epochs=10,callbacks = [tboard_callback])


# In[ ]:


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


# In[ ]:





# In[ ]:




