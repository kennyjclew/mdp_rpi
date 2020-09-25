

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


class Multithreading:

    def __init__(self):

        #instantiate all classes used
        self.arduino = Arduino()
        #self.android = Android()
        self.pc = PCserver(HOST_IP, PORT)

        #self.alldevices = [self.arduino, self.android, self.pc] 
        self.alldevices = [self.arduino, self.pc] 

        #connect all devices
        self.connect_all_devices()
        #KIV: reconnection?

        #msg queues. reads will add to the queue, writes will read from the queue
        self.msgqueue = multiprocessing.Queue()
        #each object in queue is a list [DEST_HEADER, MSG]


        #not sure if need disconnect_all for all the modules? 
        #if one never connect, then have to disconnect all? consider during testing
        self.r_arduino_thread = multiprocessing.Process(target=self.arduino_continuous_read, args=(self.msgqueue,))
        #self.r_android_thread = multiprocessing.Process(target=self.android_continuous_read, args=(self.msgqueue,))
        self.r_pc_thread = multiprocessing.Process(target=self.pc_continuous_read, args=(self.msgqueue,))
            #KIV: not sure if need args. check back later

        self.w_thread = multiprocessing.Process(target=self.write_to_device, args=(self.msgqueue,))

        #self.allthreads = [self.r_arduino_thread, self.r_android_thread, self.r_pc_thread, self.w_thread]
        self.allthreads = [self.r_arduino_thread, self.r_pc_thread, self.w_thread]

        self.start_all_threads()
        


    def connect_all_devices(self):
        try:
            self.arduino.connect()
            self.android.connect()
            self.pc.connect()
            print("all devices connected!")

        except Exception as error:
            print("error when connecting devices: " + error)


    def start_all_threads(self):
        for thread in self.allthreads:
            thread.start()


    def stop_all(self): #KIV: not sure where to call this method
        for thread in self.allthreads:
            if thread.is_alive():
                thread.terminate()

        for device in self.alldevices:
            device.disconnect() #KIV: not sure if need to disconnect_all?
        


    def arduino_continuous_read(self):
        while True:
            msg = self.arduino.read()
            #arduino only sends msges to PC. directly forward msg.

            if msg is None:
                continue
            
            self.msgqueue.put([PC_HEADER, msg]) 
            #KIV: ??nowait: raise Full exception immediately??

    def android_continuous_read(self):
        while True:
            msg = self.android.read()

            if msg is None:
                continue
            elif msg in AndroidToRPi.ANDTORPI_MESSAGES: #send to both arduino and pc
                self.msgqueue.put([PC_HEADER, msg])
            else:
                print("received invalid message from android")
            
            #need to send to arduino no matter what
            self.msgqueue.put([ARDUINO_HEADER, msg])


    def pc_continuous_read(self):
        while True:
            msg = self.pc.read()
            #pc either sends to android (only mapstring) or rpi

            if msg is None:
                continue
            elif msg[0] == PCToAndroid.MAP_STRING: #PC must send MAPSTRING with MAP_STRING as a header
                self.msgqueue.put([ANDROID_HEADER, msg])
            elif msg == PCToRPi.TAKE_PICTURE:
                #KIV: RPI TAKE PICTURE
                print("pc tells rpi to take picture")
            elif msg == PCToRPi.EXPLORATION_DONE:
                #KIV: RPI DO SOMETHING. display all images recognised?
                print("pc tells rpi that exploration done")

            else:
                print("received invalid message from PC")
                

    def write_to_device(self):
        while True:
            if not self.msgqueue.empty():
                msg = self.msgqueue.get()

                if msg[0] == ARDUINO_HEADER:
                    self.arduino.write(msg[1])
                elif msg[0] == ANDROID_HEADER:
                    self.android.write(msg[1])
                else: 
                    self.pc.write(msg[1])
                

