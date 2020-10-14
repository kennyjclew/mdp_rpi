

# create 4 threads: 
# - read messages sent from arduino, android, pc 
    #read() function only runs once. 
    #for the threaded process, must keep running read(), 
    # and only add relevant msgs (not empty) to the msg queue
# - write to arduino, android, pc


import multiprocessing

from communications import *
from Arduino_rachael import Arduino
from Android_kenny import Android
from PCserver import PCserver

#IMPORTS FOR IMAGE PROCESSING
import imagezmq
import time
from picamera import PiCamera
from picamera.array import PiRGBArray


class Multithreading:

    def __init__(self):

        #instantiate all classes used
        self.arduino = Arduino()
        self.android = Android()
        self.pc = PCserver(HOST_IP, PORT)

        self.alldevices = [self.arduino, self.android, self.pc] 

        #connect all devices
        self.connect_all_devices()
        #KIV: reconnection?

        #msg queues. reads will add to the queue, writes will read from the queue
        self.msgqueue = multiprocessing.Queue()
        self.updatesqueue = multiprocessing.Queue() #mainly for pc to android for live updating of map on android
        #each object in queue is a list [DEST_HEADER, MSG]

        
        #------IMG RECOGNITION QUEUE AND THREAD
        self.imgqueue = multiprocessing.Queue()
            #each object in queue is a list [coord, img]
        self.w_image_thread = multiprocessing.Process(target=self.write_to_imgserver, args=(self.imgqueue, self.msgqueue, ))
        #------END IMG RECOGNITION DECLARATIONS


        #not sure if need disconnect_all for all the modules? 
        #if one never connect, then have to disconnect all? consider during testing
        self.r_arduino_thread = multiprocessing.Process(target=self.arduino_continuous_read, args=(self.msgqueue,))
        self.r_android_thread = multiprocessing.Process(target=self.android_continuous_read, args=(self.msgqueue,))
        self.r_pc_thread = multiprocessing.Process(target=self.pc_continuous_read, args=(self.msgqueue, self.imgqueue, self.updatesqueue,))
            #KIV: not sure if need args. check back later

        self.w_thread = multiprocessing.Process(target=self.write_to_device, args=(self.msgqueue,))
        self.w_updates_thread = multiprocessing.Process(target=self.write_to_device, args=(self.updatesqueue,))

        self.allthreads = [self.r_arduino_thread, self.r_android_thread, self.r_pc_thread, self.w_thread, self.w_image_thread, self.w_updates_thread]

        #self.start_all_threads()

        #self.checkconnections()
        


    def connect_all_devices(self):
        try:
            for device in self.alldevices:
                device.connect()
            print("all devices connected!")

        except Exception as error:
            print("error when connecting devices: " + str(error))


    def start_all_threads(self):
        for thread in self.allthreads:
            thread.start()
        
        self.checkconnections()

    def stop_all(self): #KIV: not sure where to call this method
        for thread in self.allthreads:
            if thread.is_alive():
                thread.terminate()

        # for device in self.alldevices:
        #     device.disconnect() #KIV: not sure if need to disconnect_all?
        
        self.arduino.disconnect()
        self.android.disconnect_all()
        self.pc.disconnect_both()
        

    def checkconnections(self):
        while True:
            #assumes that process dies when target dies?????
            arduinoalive = self.r_arduino_thread.is_alive()
            androidalive = self.r_android_thread.is_alive()
            pcalive = self.r_android_thread.is_alive()

            if (arduinoalive and androidalive and pcalive):
                continue
            else:
                try:
                    #stop all write processes so robot pauses
                    self.w_thread.terminate()
                    self.w_updates_thread.terminate()

                    if not arduinoalive:
                        print("restarting connection with arduino")
                        self.arduino.disconnect()
                        self.r_arduino_thread.terminate()
                        self.arduino.connect()
                        self.r_arduino_thread = multiprocessing.Process(target=self.arduino_continuous_read, args=(self.msgqueue,))
                        self.r_arduino_thread.start()
                    elif not (androidalive and self.android.checkconnections()):
                        print("restarting connection with android")
                        self.android.disconnect()
                        self.r_android_thread.terminate()
                        self.android.connect()  
                        self.r_android_thread = multiprocessing.Process(target=self.android_continuous_read, args=(self.msgqueue,))
                        self.r_android_thread.start()
                    elif not pcalive:
                        print("restarting connection with pc")
                        self.pc.disconnect()
                        self.r_pc_thread.terminate()
                        self.pc.connect()
                        self.r_pc_thread = multiprocessing.Process(target=self.pc_continuous_read, args=(self.msgqueue, self.imgqueue, self.updatesqueue,))
                        self.r_pc_thread.start()
                    self.w_thread = multiprocessing.Process(target=self.write_to_device, args=(self.msgqueue,))
                    self.w_updates_thread = multiprocessing.Process(target=self.write_to_device, args=(self.updatesqueue,))
                    self.w_thread.start()
                    self.w_updates_thread.start()
                except Exception as e:
                    print("error while reconnecting: " + e)
                    continue


    def arduino_continuous_read(self, msgqueue):
        while True:
            msg = self.arduino.read()
            #arduino only sends msges to PC. directly forward msg.

            if msg is None:
                continue
            
            msgqueue.put([PC_HEADER, msg]) 
            #KIV: ??nowait: raise Full exception immediately??

    def android_continuous_read(self, msgqueue):
        while True:
            msg = self.android.read()

            if msg is None:
                continue
            elif (msg[:8] == AndroidToPC.WAYPOINT): #send waypoints to pc
                waypoint_coord = msg[10:-1]
                print("Hello waypoint" + waypoint_coord)
                msgqueue.put([PC_HEADER, waypoint_coord])
            elif msg in AndroidToArduino.ANDTOARD_MESSAGES: #AND sends WSAD, calibrate to ARD
                msgqueue.put([ARDUINO_HEADER, msg])
            elif msg in AndroidToRPi.ANDTORPI_MESSAGES: #send to both arduino and pc
                msgqueue.put([PC_HEADER, msg])
                msgqueue.put([ARDUINO_HEADER, msg])
            else:
                print("received invalid message from android")


    def pc_continuous_read(self, msgqueue, imgqueue, updatesqueue):
        while True:
            msg = self.pc.read()
            #pc either sends to android (only mapstring) or rpi
            msg_list = msg.splitlines()
            for msg in msg_list:
                print("READING PC "+ msg)
                if msg is None:
                    continue
                elif msg[0] == PCToAndroid.MAP_STRING: #PC must send MAPSTRING with MAP_STRING as a header
                    updatesqueue.put([ANDROID_HEADER, msg[1:]])
                elif msg[:2] == PCToAndroid.ROBOT_POSITION:
                    updatesqueue.put([ANDROID_HEADER, msg[2:]])
                elif msg[:2] == PCToRPi.TAKE_PICTURE:
                    print("pc tells rpi to take picture")
                    img = self.take_picture(imgqueue)
                    imgqueue.put([msg[3:], img])
                elif msg == PCToRPi.EXPLORATION_DONE:
                    #KIV: RPI DO SOMETHING. display all images recognised?
                    print("pc tells rpi that exploration done")
                else:
                    msgqueue.put([ARDUINO_HEADER, msg])
                    print("msg from PC forwarding to arduino")
                

    def write_to_device(self, msgqueue):
        while True:
            if not msgqueue.empty():
                msg = msgqueue.get()

                if msg[0] == ARDUINO_HEADER:
                    self.arduino.write(msg[1])
                elif msg[0] == ANDROID_HEADER:
                    self.android.write(msg[1])
                else: 
                    self.pc.write(msg[1])
                

    #------IMAGE RECOGNITION METHODS------
    def take_picture(self, imgqueue):
        try:
            rpicamera = PiCamera(resolution=(1920,1088))
            rpicamera.hflip = True
            outputtype = PiRGBArray(rpicamera)
            #time.sleep(0.1) #camera may need to warm up? KIV

            rpicamera.capture(outputtype, format="bgr")
            imgtaken = outputtype.array
            print("Image taken")
            rpicamera.close()

            # to gather training images
            # os.system("raspistill -o images/test"+
            # str(start_time.strftime("%d%m%H%M%S"))+".png -w 1920 -h 1080 -q 100")
        
        except Exception as error:
            print('Taking picture failed: ' + str(error))
        
        return imgtaken

    def write_to_imgserver(self, imgqueue, msgqueue):
        image_sender = imagezmq.ImageSender(connect_to="tcp://192.168.21.31:5555") #bryna
        #image_sender = imagezmq.ImageSender(connect_to="tcp://192.168.21.35:5555") #kenny
        print("connected to image server")
        while True:
            if not imgqueue.empty():
                imgwcoord = imgqueue.get()
                imgserverreply = image_sender.send_image(imgwcoord[0], imgwcoord[1]) 
                #assume server will reply with 1. img detected, 2. coordinates 
                # if imgserverreply is not None:
                #     msgqueue.put([ANDROID_HEADER, imgserverreply])
                print("image recognition complete")
