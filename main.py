from os import listdir

import numpy as np
import pandas as pd
import tensorflow as tf
import os
from PIL import Image
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
import time, datetime
import kagglehub


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
        plt.savefig(figure_dir+"/history")
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




