# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 20:08:08 2020

@author: Rachael
"""

import math
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import glob 
import os
from tqdm import tqdm
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, BatchNormalization, Flatten, MaxPooling2D, Lambda, ReLU
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from numpy import expand_dims
import datetime as dt

def getArchitecture():
        input_layer = Input(shape=(64,64, 3))
        x = Conv2D(32, (3,3), padding="same")(input_layer)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(32, (3,3), padding="same")(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D()(x)

        x = Conv2D(64, (5,5), padding="same")(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(64, (5,5), padding="same")(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D()(x)

        x = Conv2D(128, (5,5), padding="same")(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(128, (5,5), padding="same")(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D()(x)

        x = Conv2D(256, (5,5), padding="same")(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(256, (5,5), padding="same")(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D()(x)

        x = Flatten()(x)
        x = Dense(512, activation="relu")(x)
        output = Dense(15, activation="softmax")(x)

        # Connect the inputs with the outputs
        cnn = Model(inputs=input_layer,outputs=output)
        cnn.compile(loss="categorical_crossentropy",optimizer=Adam(lr=0.00006, decay=1e-6),metrics=['acc'])
        return cnn
    
#Load Architecture
model = getArchitecture()
#Load Model
model.load_weights("C:/Users/bryna/OneDrive/Documents/GitHub/mdp_rpi/image_processing_server/model_new.h5") # CHANGE PATH

def resizeImage(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation = inter)
    return resized


def getBoundingBox(contours):
    box = []
    # sort size in descending order
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        #print(area)
        if area < 200: # area of img is too small, skip
            break
        elif area > 12000: # area of img is too big, skip
            continue
        rect = cv2.boundingRect(contour)
        x, y, w, h = rect
        if w > 2*h: # img is far from robot 
            continue 
        box.append(np.array(rect))
    return np.array(box)

def checkBrightness(target):
    img = resizeImage(target, height = 900)
    img_dot = img

    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    y,x,z = img.shape #height, width of image
    # Check light intensity
    l_blur = cv2.GaussianBlur(l, (11, 11), 5)
    maxval = []
    count_percent = 3 # percent of total image
    count_percent = count_percent/100
    row_percent = int(count_percent*x) # 1% of total pixels widthwise
    column_percent = int(count_percent*y) # 1% of total pizel height wise
    for i in range(1,x-1):
        if i%row_percent == 0:
            for j in range(1, y-1):
                if j%column_percent == 0:
                    pix_cord = (i,j)
cv2.circle(img_dot, (int(i), int(j)), 5, (0, 255, 0), 2)
                    img_segment = l_blur[i:i+3, j:j+3]
                    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(img_segment)
                    maxval.append(maxVal)

    avg_maxval = round(sum(maxval) / len(maxval))
    return avg_maxval


def runAnalysis(img_path):
    imageCounter = 0
    img = cv2.imread(img_path)
    img = resizeImage(img, height = 340)
    crop = img[100:, :]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 0)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    box = getBoundingBox(contours)
    CATEGORIES  = ['Circle', 'Red Arrow', 'Blue Arrow', '8', '6','0','7','W','V','White Arrow', 'X', 'Y', 'Yellow Arrow', 'Z','9']
    predictions = []
    boxes = []
    if len(box) > 0:
        for b in box:
            x, y, w, h = b
            target = crop[y:y+h, x:x+w] # crop out the bounding box image
            if (checkBrightness(target)<=30):
                continue
            filename = r'C:/Users/bryna/Documents/UNIVERSITY/YEAR 3/SEM 1/Multidisciplinary Project/RPi/img recognition/server test/processed images/target/' + dt.datetime.now().strftime("%Y%m%d-%H%M%S") + '.jpg' #CHANGE PATH
            cv2.imwrite(filename, target)            
            test = cv2.resize(target, (64, 64), interpolation=cv2.INTER_CUBIC)  # resize to normalize data size
            test = np.reshape(test, newshape=(-1, 64, 64,3))
            predictions.append(model.predict(test/255))
            boxes.append(box)
            
    bestResults = [None, 0, None] # label, prob, box
    
    # To get the highest prob
    i = 0
    for pred in predictions:
            prob = np.max(pred, axis=1)
            classLabel = np.argmax(pred, axis=1)

            if prob > 0.95 and prob > bestResults[1]:
                bestResults[1] = prob
                classLabel = np.argmax(pred, axis=1)
                bestResults[0] = CATEGORIES[classLabel[0]]
                bestResults[2] = box[i]
            i = i+1   
    
    print("\n")
    print("Best Result")
    print(bestResults[1],bestResults[0])
    
    #Draw coutour to original image
    x, y, w, h = bestResults[2]
    y = y+100
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
    text = "{}: {:.2f}%".format(bestResults[0].upper(), (bestResults[1]*100)[0])
    cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    filename = r'C:/Users/bryna/Documents/UNIVERSITY/YEAR 3/SEM 1/Multidisciplinary Project/RPi/img recognition/server test/processed images/' + bestResults[0] + str(imageCounter) + '_' + dt.datetime.now().strftime("%Y%m%d-%H%M%S") + '.jpg' #CHANGE PATH
    cv2.imwrite(filename, img)
    imageCounter = imageCounter + 1