from os import listdir

import numpy as np
import pandas as pd
import tensorflow as tf
import os
from PIL import Image
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, InputLayer
import time, datetime
import kagglehub
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=15,       # 15 dereceye kadar döndürme
    width_shift_range=0.1,   # Genişlik kaydırma (%10)
    height_shift_range=0.1,  # Yükseklik kaydırma (%10)
    shear_range=0.2,         # Kesme dönüşümü
    zoom_range=0.2,          # Rastgele yakınlaştırma
    brightness_range=[0.8, 1.2],  # Parlaklık değişimi
    horizontal_flip=False,   # Trafik işaretleri yönlü olduğu için kapalı
    fill_mode='nearest'      # Kenarları doldurma yöntemi
)

def date_time(x):
    if x == 1:
        return 'Timestamp : {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    elif x == 2:
        return 'Timestamp : {:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now())
    elif x == 3:
        return 'Date now: %s' % datetime.datetime.now()
    elif x == 4:
        return 'Date today: %s' % datetime.date.today()


def plot_performance(history=None, figure_dir=None, ylim_pad=[0,0]):
    xlabel = "epoch"
    legends = ["Training", "Validation"]
    plt.figure(figsize=(20,5))
    y1 = history.history['accuracy']
    y2 = history.history['val_accuracy']

    min_y = min(min(y1), min(y2)) - ylim_pad[0]
    max_y = max(max(y1), max(y2)) + ylim_pad[0]

    plt.subplot(121)

    plt.plot(y1)
    plt.plot(y2)
    plt.xlabel(xlabel, fontsize=15)
    plt.title("Model Accuracy\n"+date_time(1), fontsize=20)
    plt.ylim(min_y, max_y)
    plt.legend(legends)
    plt.ylabel("Accuracy",fontsize=15)
    plt.grid(True)

    y1 = history.history['loss']
    y2 = history.history['val_loss']

    min_y = min(min(y1), min(y2)) - ylim_pad[1]
    max_y = max(max(y1), max(y2)) + ylim_pad[1]

    plt.subplot(122)
    plt.plot(y1)
    plt.plot(y2)
    plt.xlabel(xlabel, fontsize=15)
    plt.title("Model Loss\n"+date_time(1), fontsize=20)
    plt.ylim(min_y, max_y)
    plt.ylabel("Loss",fontsize=15)
    plt.grid(True)
    plt.legend(legends)

    if figure_dir:
        path = os.path.join(figure_dir, "history.png")
        print(path)
        plt.savefig(path)
    plt.show()


path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
print(path)
data = []
labels = []
NUM_CLASSES = 43

for i in range(NUM_CLASSES):
    train_path = os.path.join(path, 'train', str(i))
    for img_path in listdir(train_path):
        try:
            image = Image.open(os.path.join(train_path, img_path))
            image = image.resize((30,30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except:
            print(f"Error loading {i}/{img_path} image")

data = np.array(data)
labels = np.array(labels)
print(data.shape, labels.shape)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)
train_generator = datagen.flow(X_train, y_train, batch_size=64, shuffle=True)

model = Sequential()
model.add(InputLayer(input_shape=X_train.shape[1:]))
model.add(Conv2D(32, (5,5), activation='relu'))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.15))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()


with tf.device('/GPU:0'):
    epochs = 30
    history1 = model.fit(train_generator, epochs=epochs, validation_data=(X_test, y_test))

plot_performance(history1, figure_dir=os.getcwd(), ylim_pad=[0,0])

y_test = pd.read_csv(os.path.join(path, 'Test.csv'))
test_labels = y_test["ClassId"].values
test_imgs = y_test["Path"].values
test_data = []

with tf.device('/GPU:0'):
    for img_path in test_imgs:
        img = Image.open(os.path.join(path, img_path))
        img = img.resize((30,30))
        test_data.append(np.array(img))


X_test = np.array(test_data)

with tf.device('/GPU:0'):
    predictions = model.predict(X_test)
    predictions = predictions.argmax(axis=-1)

print(accuracy_score(predictions, test_labels))
