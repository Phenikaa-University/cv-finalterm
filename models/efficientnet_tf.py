"""Implementation of EfficientNet in TensorFlow."""
import tensorflow as tf
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization
from keras.optimizers import Adamax
from keras.metrics import categorical_crossentropy
from keras.models import Model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

img_size = (500, 500)
image_lst = []
image_dir = 'demo/images'
n_label = 8

# Load in the images
for filepath in os.listdir('demo/images'):
    img = cv2.imread(os.path.join(image_dir,filepath))
    img = cv2.resize(img, img_size)
    image_lst.append(img)
    
image_array = np.stack(image_lst, axis=0)
x = image_array

df = pd.read_excel('demo/data_refine_all_errors_filtered.xlsx')
label_detail = df['label_detail']
label_str_set = ['0','1','2','3','4','5','6','7']

def convert_label(label_str, label_str_set):
    labels = []
    for char in label_str:
        if char in label_str_set:
            labels.append(int(char))
    return labels

labels_lst = []
for label_str in label_detail:
    labels = convert_label(label_str, label_str_set)
    labels_lst.append(labels)

labels_array = np.zeros((len(labels_lst),8))
for i in range(len(labels_lst)):
    for label in labels_lst[i]:
        labels_array[i][label] = 1.0
        
x_train, x_test, y_train, y_test = train_test_split(x, labels_array, test_size=0.2, random_state=42)


# Xây dựng mô hình
# Build model
base_model = tf.keras.applications.efficientnet.EfficientNetB3(include_top=False,
                                                              weights="imagenet",
                                                              input_shape = (img_size[0], img_size[1], 3),
                                                                pooling = "max"
                                                              )

# base_model = tf.keras.applications.resnet50.ResNet50(include_top=False,
#                                                               weights="imagenet",
#                                                               input_shape = (img_size[0], img_size[1], 3),
#                                                                 pooling = "max"
#                                                               )

base_model.trainable = True

x = base_model.output
x = BatchNormalization() (x)
x = Dense(1024, activation = "relu") (x)
x = Dropout(0.3) (x)
x = Dense(512, activation = "relu") (x)
x = Dropout(0.3) (x)
x = Dense(128, activation = "relu") (x)
x = Dropout(0.3) (x)
outputs = Dense(8, activation = "sigmoid") (x)

model = Model(inputs = base_model.input, outputs = outputs)

# Define your optimizer with a learning rate
learning_rate = 0.001
optimizer_1 = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Compile mô hình
model.compile(optimizer=optimizer_1,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(x_train, y_train, epochs=1, batch_size=2, validation_data=(x_test, y_test))

# Đánh giá mô hình trên dữ liệu kiểm tra
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Độ chính xác trên dữ liệu kiểm tra:', test_acc)

# x_test = x_train
# y_test = y_train

y_pred = model.predict(x_test)
y_pred[np.where(y_pred>0.5)] = 1
y_pred[np.where(y_pred<=0.5)] = 0

diff = abs(y_pred-y_test)
acc = sum(sum(diff))/len(y_pred) / n_label
print('Acruracy: ', acc)