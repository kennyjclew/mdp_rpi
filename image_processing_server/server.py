"""
This file contains the ImageProcessingServer class.
"""
import os
import shutil
import sys
from datetime import datetime

import cv2
import imutils
import numpy as np
# import tensorflow as tf

from config import *
from image_receiver import custom_imagezmq as imagezmq
import time
import imgrecognTest
# from utils import label_map_util
# from utils import visualization_utils as vis_util

#FOR IMAGE SLICING
from scipy import misc
import imageio


# sys.path.append("..")

# Grab path to current working directory
# cwd_path = os.getcwd()


class ImageProcessingServer:
    def __init__(self):
        
        # # initialize the ImageHub object
        self.image_hub = imagezmq.CustomImageHub()
        self.CATEGORIES  = {'White Arrow': "1",
                        'Blue Arrow': "2",
                        'Yellow Arrow': "3",
                        'Red Arrow': "4",
                        'Circle': "5",
                        '6': "6",
                        '7': "7", 
                        '8': "8",
                        '9': "9",
                        '0': "10", 
                        'V': "11", 
                        'W': "12",
                        'X': "13", 
                        'Y': "14",  
                        'Z': "15"}

        self.COORDINATES = {"1": 0,
                            "2": 0,
                            "3": 0,
                            "4": 0,
                            "5": 0,
                            "6": 0,
                            "7": 0,
                            "8": 0,
                            "9": 0,
                            "10": 0,
                            "11": 0,
                            "12": 0,
                            "13": 0,
                            "14": 0,
                            "15": 0}

    def start(self):
        print('\nStarted image processing server.\n')
        
        
        while True:
            print('Waiting for image from RPi...')

            # receive RPi name and frame from the RPi and acknowledge the receipt
            coord , frame = self.image_hub.recv_image() #coord in format y(row)|x(col)
            print('\nConnected and received frame at time: ' + str(datetime.now()) + " at coordinate: " + coord)
            
            coordlist = coord.split("|")
            leftcoord = str("(" + coordlist[0] + " , " + coordlist[1] + ")")
            middlecoord = str("(" + coordlist[2] + " , " + coordlist[3] + ")")
            rightcoord = str("(" + coordlist[4] + " , " + coordlist[5] + ")")
            print("left: " + leftcoord + "; middle: " + middlecoord + "; right: " + rightcoord)

            frame = imutils.resize(frame, width=IMAGE_WIDTH)
            
            datetimestring = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

            baseurl = "C:/Users/bryna/Documents/UNIVERSITY/YEAR 3/SEM 1/Multidisciplinary Project/RPi/img recognition/server test"
            # save raw image
            save_success = cv2.imwrite(baseurl + "/trainingimages/FULL" + datetimestring + ".jpg", frame)
            print('save', "test.jpg", 'successful?', save_success)

            #fullResult = imgrecognTest.runAnalysis(baseurl + "/trainingimages/FULL" + datetimestring + ".jpg")
            #print("\n fullResult: " + fullResult + " between " + leftcoord + " and " + rightcoord + "\n")

            test3tuple = cut_image(self, baseurl + "/trainingimages/FULL" + datetimestring + ".jpg", baseurl + "/SLICED_IMAGES/")

            leftCoordResult = ""
            middleCoordResult = ""
            rightCoordResult = ""
            
            

            leftResult = imgrecognTest.runAnalysis(baseurl + "/SLICED_IMAGES/" + test3tuple[0])
            if leftResult is not None:
                print("\n LeftResult: " + leftResult + " at " + leftcoord + "\n")
                leftCoordResult = ('{"image":[' + self.CATEGORIES[leftResult] + "," + coordlist[1] + "," + coordlist[0] + "]}")
                if self.COORDINATES[self.CATEGORIES[leftResult]] == 0:
                    print("self.COORDINATES[self.CATEGORIES[leftResult]] EQUAL 0")
                    for i in self.COORDINATES:
                        if self.COORDINATES[i] == "(" + coordlist[0] + "," + coordlist[1] + ")":
                            print("SAME COORD DETECTED FOR LEFT")
                            leftCoordResult = ""
                    if leftCoordResult != "":
                        self.COORDINATES[self.CATEGORIES[leftResult]] = "(" + coordlist[0] + "," + coordlist[1] + ")" 
                    
                else:
                    leftCoordResult = ""
    
            middleResult = imgrecognTest.runAnalysis(baseurl + "/SLICED_IMAGES/" + test3tuple[1])
            if middleResult is not None:
                print("\n middleResult: " + middleResult + " at " + middlecoord + "\n")
                middleCoordResult = ('{"image":[' + self.CATEGORIES[middleResult] + "," + coordlist[3] + "," + coordlist[2] + "]}")
                if self.COORDINATES[self.CATEGORIES[middleResult]] == 0:
                    print("self.COORDINATES[self.CATEGORIES[middleResult]] EQUAL 0")
                    for i in self.COORDINATES:
                        if self.COORDINATES[i] == "(" + coordlist[2] + "," + coordlist[3] + ")":
                            print("SAME COORD DETECTED FOR middle")
                            middleCoordResult = ""
                    if middleCoordResult != "":
                        self.COORDINATES[self.CATEGORIES[middleResult]] = "(" + coordlist[2] + "," + coordlist[3] + ")"   
                else:
                    middleCoordResult = ""
    
            rightResult = imgrecognTest.runAnalysis(baseurl + "/SLICED_IMAGES/" + test3tuple[2])
            if rightResult is not None:
                print("\n rightResult: " + rightResult + " at " + rightcoord + "\n")
                rightCoordResult = ('{"image":[' + self.CATEGORIES[rightResult] + "," + coordlist[5] + "," + coordlist[4] + "]}")
                if self.COORDINATES[self.CATEGORIES[rightResult]] == 0:
                    print("self.COORDINATES[self.CATEGORIES[rightResult]] EQUAL 0")
                    for i in self.COORDINATES:
                        if self.COORDINATES[i] == "(" + coordlist[4] + "," + coordlist[5] + ")":
                            print("SAME COORD DETECTED FOR right")
                            rightCoordResult = ""
                    if rightCoordResult != "":
                        self.COORDINATES[self.CATEGORIES[rightResult]] = "(" + coordlist[4] + "," + coordlist[5] + ")" 
                else:
                    rightCoordResult = ""
    

            self.image_hub.send_reply(leftCoordResult + "|" + middleCoordResult + "|" + rightCoordResult)

            imgrecognTest.stitchAllImage()

#FOR IMAGE SLICING
def cut_image(self, img_path, save_path): 
    # Read the image
    img = imageio.imread(img_path)
    height, width, _ = img.shape
    # print(img.shape)

    # Cut the image in half
    width_cutoff = width // 3
    s1 = img[:, :width_cutoff]
    s2 = img[:,width_cutoff: width_cutoff*2]
    s3 = img[:, width_cutoff*2:]
    #s3 = img[width_cutoff*2:,:]

    datetimestring = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Save each half
    imageio.imsave(save_path+ datetimestring + "_face1.jpg", s1)
    imageio.imsave(save_path+ datetimestring + "_face2.jpg", s2)
    imageio.imsave(save_path+ datetimestring + "_face3.jpg", s3)

    return (datetimestring+"_face1.jpg", datetimestring+"_face2.jpg", datetimestring+"_face3.jpg")
