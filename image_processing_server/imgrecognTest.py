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
# example of horizontal flip image augmentation
from numpy import expand_dims

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
        if area < 200: # too small, skip
            break
        elif area > 12000: # too big, skip
            continue
        rect = cv2.boundingRect(contour)
        x, y, w, h = rect
        #if w < 20 or h < 20: 
            #break
        if w > 2*h: # w too big, 
            continue 
        box.append(np.array(rect))
    return np.array(box)
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
            test = cv2.resize(target, (64, 64), interpolation=cv2.INTER_CUBIC)  # resize to normalize data size
            test = np.reshape(test, newshape=(-1, 64, 64,3))
            predictions.append(model.predict(test/255))
            boxes.append(box)
            
    bestResults = [None, 0, None] # label, prob, box
    
    #get the best probability
    i = 0
    for pred in predictions:
            prob = np.max(pred, axis=1)
            classLabel = np.argmax(pred, axis=1)
            if prob > 0.80 and prob > bestResults[1]:
                bestResults[1] = prob
                classLabel = np.argmax(pred, axis=1)
                bestResults[0] = CATEGORIES[classLabel[0]]
                bestResults[2] = box[i]
            i = i+1   
    print("Best Result")
    print(bestResults[1],bestResults[0])
    
    #Draw coutour to original image
    x, y, w, h = bestResults[2]
    y = y+100
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
    text = "{}: {:.2f}%".format(bestResults[0].upper(), (bestResults[1]*100)[0])
    cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    filename = r'C:/Users/bryna/Documents/UNIVERSITY/YEAR 3/SEM 1/Multidisciplinary Project/RPi/img recognition/server test/processed images/' + bestResults[0] + str(imageCounter) + '.jpg' #CHANGE PATH
    cv2.imwrite(filename, img)
    imageCounter = imageCounter + 1

# import tensorflow as tf

# from tensorflow.keras import datasets, layers, models
# # import matplotlib.pyplot as plt
# import cv2
# import glob 
# import numpy as np
# import PIL
# import PIL.Image
# import os
# from tqdm import tqdm
# from keras.models import Model
# from keras.layers import Input, Dense, Conv2D, BatchNormalization, Flatten, MaxPooling2D, Lambda, ReLU
# from keras.optimizers import Adam
# from keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# from keras.utils import to_categorical
# # example of horizontal flip image augmentation
# from numpy import expand_dims
# from tensorflow import keras
# import datetime as dt

# # # mapping for class id
# # class_mapping = {'up':1, 'down':2, 'right':3, 'left':4, 'circle':5, 'one':6, 'two':7,
# #                 'three':8, 'four':9, 'five':10, 'a':11, 'b':12, 'c':13, 'd':14, 'e':15}

# def getArchitecture():
#         input_layer = Input(shape=(64,64, 3))
#         x = Conv2D(32, (3,3), padding="same")(input_layer)
#         x = BatchNormalization()(x)
#         x = ReLU()(x)
#         x = Conv2D(32, (3,3), padding="same")(x)
#         x = BatchNormalization()(x)
#         x = ReLU()(x)
#         x = MaxPooling2D()(x)

#         x = Conv2D(64, (5,5), padding="same")(x)
#         x = BatchNormalization()(x)
#         x = ReLU()(x)
#         x = Conv2D(64, (5,5), padding="same")(x)
#         x = BatchNormalization()(x)
#         x = ReLU()(x)
#         x = MaxPooling2D()(x)

#         x = Conv2D(128, (5,5), padding="same")(x)
#         x = BatchNormalization()(x)
#         x = ReLU()(x)
#         x = Conv2D(128, (5,5), padding="same")(x)
#         x = BatchNormalization()(x)
#         x = ReLU()(x)
#         x = MaxPooling2D()(x)

#         x = Conv2D(256, (5,5), padding="same")(x)
#         x = BatchNormalization()(x)
#         x = ReLU()(x)
#         x = Conv2D(256, (5,5), padding="same")(x)
#         x = BatchNormalization()(x)
#         x = ReLU()(x)
#         x = MaxPooling2D()(x)

#         x = Flatten()(x)
#         x = Dense(512, activation="relu")(x)
#         output = Dense(15, activation="softmax")(x)

#         # Connect the inputs with the outputs
#         cnn = Model(inputs=input_layer,outputs=output)
#         cnn.compile(loss="categorical_crossentropy",optimizer=Adam(lr=0.00006, decay=1e-6),metrics=['accuracy'])
#         # cnn.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#         return cnn
    
# def getBoundingBox(contours):
#     box = []
    
#     #test to find out the delim for area
    
#     # sort size in descending order
#     for contour in sorted(contours, key=cv2.contourArea, reverse=True):
#         area = cv2.contourArea(contour)
#         print(area)
#         if area < 200: # too small, exit from iteration
#             break
#         elif area > 12000: # too big, skip
#             continue
#         rect = cv2.boundingRect(contour)
#         x, y, w, h = rect
#         if w < 20 or h < 20: # too small, ignore
#             break
#         if w > 2*h: # width too big, unlikely our target
#             continue
#         box.append(np.array(rect))
#     return np.array(box)

# def runAnalysis(img_path):
#     # Load Model and Image
#     CATEGORIES  = ['Circle', 'Red Arrow', 'Blue Arrow', '8', '6','0','7','W','V','White Arrow', 'X', 'Y', 'Yellow Arrow', 'Z']
#     modeltest = getArchitecture()
#     modeltest.load_weights("C:/Users/bryna/OneDrive/Documents/GitHub/mdp_rpi/image_processing_server/model.h5") # load saved weights
#     imgtest = cv2.imread(img_path)  # convert to array
#     imgtest = cv2.resize(imgtest, (320, 240)) #240, 320
#     # test to find out how much to crop
#     crop = imgtest[100:, :] # crop the frame
#     gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) # convert to grayscale
#     blur = cv2.GaussianBlur(gray, (5,5), 0) # blur the background
#     threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 0)
#     contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     box = getBoundingBox(contours)
#     # cv2.imshow('Crop', crop)
#     # cv2.waitKey(0)
        
#     predictions = []
#     boxes=[]
#     if len(box) > 0:
#         for b in box:
#             x, y, w, h = b
#             target = crop[y:y+h, x:x+w] # crop out the bounding box image
#             cv2.imshow('Target', target)
#             cv2.waitKey(0)
#             # resized = cv2.resize(target, dsize=(modeltest.dim, modeltest.dim), interpolation=cv2.INTER_CUBIC) # resize to match CNN input size
#             # normed = resized/255 # normalize before pass through CNN
#             img = cv2.resize(target, (64, 64))
#             img = np.reshape(img, newshape=(1, 64, 64,3)) 
#             # img = img/255.0 # normalize
                
#             # Predict Image
#             result = modeltest.predict(img)
#             class_labels = np.argmax(result, axis=1) # assuming you have n-by-5 class_prob
#             class_labels=class_labels.ravel()
#             label = CATEGORIES[class_labels[0]]
#             # scores = modeltest.evaluate(imgtest, verbose=0)
#             print('Predict: ' + label)
#             print(np.max(result, axis=1))
#             # print("Accuracy: %.2f%%" % (scores[1]*100))
#             predictions.append(modeltest.predict(img))
#             boxes.append(box)
#             # plt.imshow(cv2.cvtColor(target, cv2.COLOR_BGR2RGB))
#             # as opencv loads in BGR format by default, we want to show it in RGB.
#             # plt.show()
                    
#     if len(predictions) == 0:
#         # return -1, None, None, 
#         return print('No Predictions Done')
        
#     bestResults = [None, 0, None] # class label, prob, box
        
#     i = 0
#     # Go through all the possible bounding rectangles
#     # and find the highest prob that are above defined threshold
#     for p in predictions:
#         prob = np.max(p, axis=1) # assum you have n-by-5 class_prob
#         if prob > 0.65 and prob > bestResults[1]:
#             bestResults[1] = prob
#             classlabel = np.argmax(result, axis=1) # assuming you have n-by-5 class_prob
#             bestResults[0] = CATEGORIES[class_labels[0]]
#             bestResults[2] = box[i]
#         i += 1
                        
#     # below occurs when every prediction fails the threshold comparison
#     if bestResults[1] == 0:
#         # return -1, None, None, None
#         return print('No Matched Found')
        
#     x, y, w, h = bestResults[2]
#     y = y+100
        
#     # Draw the bounding rectangle on the image
#     cv2.rectangle(imgtest,(x,y),(x+w,y+h),(0,255,0),1)
#     text = "{}: {:.2f}%".format(bestResults[0].upper(), (bestResults[1]*100)[0])
#     # Write the class label of the image
#     cv2.putText(imgtest, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
#     # cv2.putText(imgtest, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
#     # Saving processed image
#     filename = 'C:/Users/bryna/Documents/UNIVERSITY/YEAR 3/SEM 1/Multidisciplinary Project/RPi/img recognition/server test/processed images/' + label + '_' + dt.datetime.now().strftime("%Y%m%d") + '.JPG'
#     cv2.imwrite(filename, imgtest)
