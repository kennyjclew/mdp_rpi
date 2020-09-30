

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
        self.android = Android()
        self.pc = PCserver(HOST_IP, PORT)

        self.alldevices = [self.arduino, self.android, self.pc] 

        #connect all devices
        self.connect_all_devices()
        #KIV: reconnection?

        #msg queues. reads will add to the queue, writes will read from the queue
        self.msgqueue = multiprocessing.Queue()
        #each object in queue is a list [DEST_HEADER, MSG]


        #not sure if need disconnect_all for all the modules? 
        #if one never connect, then have to disconnect all? consider during testing
        self.r_arduino_thread = multiprocessing.Process(target=self.arduino_continuous_read, args=(self.msgqueue,))
        self.r_android_thread = multiprocessing.Process(target=self.android_continuous_read, args=(self.msgqueue,))
        self.r_pc_thread = multiprocessing.Process(target=self.pc_continuous_read, args=(self.msgqueue,))
            #KIV: not sure if need args. check back later

        self.w_thread = multiprocessing.Process(target=self.write_to_device, args=(self.msgqueue,))

        self.allthreads = [self.r_arduino_thread, self.r_android_thread, self.r_pc_thread, self.w_thread]

        self.start_all_threads()

        self.checkconnections()
        


    def connect_all_devices(self):
        try:
            self.arduino.connect()
            self.android.connect()
            self.pc.connect()
            print("all devices connected!")

        except Exception as error:
            print("error when connecting devices: " + str(error))


    def start_all_threads(self):
        for thread in self.allthreads:
            thread.start()


    def stop_all(self): #KIV: not sure where to call this method
        for thread in self.allthreads:
            if thread.is_alive():
                thread.terminate()

        for device in self.alldevices:
            device.disconnect() #KIV: not sure if need to disconnect_all?
        

    def checkconnections(self):
        while True:
            #assumes that process dies when target dies?????
            if not self.r_arduino_thread.is_alive():
                print("restarting connection with arduino")
                self.arduino.disconnect()
                try:
                    self.arduino.connect()
                except Exception as error:
                    continue
                self.r_arduino_thread = multiprocessing.Process(target=self.arduino_continuous_read, args=(self.msgqueue,))
                self.r_arduino_thread.start()
            elif not self.r_android_thread.is_alive():
                print("restarting connection with android")
                self.android.disconnect()
                try:
                    self.android.connect()
                except Exception as error:
                    continue
                self.r_android_thread = multiprocessing.Process(target=self.android_continuous_read, args=(self.msgqueue,))
                self.r_android_thread.start()
            elif not self.r_pc_thread.is_alive():
                print("restarting connection with pc")
                self.pc.disconnect()
                try:
                    self.pc.connect()
                except Exception as error:
                    continue
                self.r_pc_thread = multiprocessing.Process(target=self.pc_continuous_read, args=(self.msgqueue,))
                self.r_pc_thread.start()
            
        #not sure why they reconnected the write threads also? kiv if need to do.


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
            elif msg in AndroidToRPi.ANDTORPI_MESSAGES: #send to both arduino and pc
                msgqueue.put([PC_HEADER, msg])
            else:
                print("message from android to be forwarded to only arduino")
            
            #need to send to arduino no matter what
            msgqueue.put([ARDUINO_HEADER, msg])


    def pc_continuous_read(self, msgqueue):
        while True:
            msg = self.pc.read()
            #pc either sends to android (only mapstring) or rpi
            msg_list = msg.splitlines()
            for msg in msg_list:
                if msg is None:
                    continue
                elif msg[0] == PCToAndroid.MAP_STRING: #PC must send MAPSTRING with MAP_STRING as a header
                    msgqueue.put([ANDROID_HEADER, msg])
                elif msg == PCToRPi.TAKE_PICTURE:
                    #KIV: RPI TAKE PICTURE
                    print("pc tells rpi to take picture")
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
                

