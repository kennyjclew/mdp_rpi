import serial
import time

class Arduino:
    #information about Arduino
    def __init__(self):
        # self.serial_port = "/dev/ttyUSB0"
        self.serial_port = '/dev/ttyACM0'
        self.baud_rate = 115200
        self.connection = None

    def connect(self):
        connected = False
        while not connected:
            try:
                print('Connecting to Arduino...')

                if self.connection is None:
                    self.connection = serial.Serial(self.serial_port, self.baud_rate)
                #if self.connection is not None, exception will be thrown

                #after attempting to connect, we double check if it is connected
                if self.connection is not None: # It is connected to at least 1 of the Arduino
                    print('Connection with Arduino ' + self.connection.name + ' SUCCESSFUL')
                    connected = True

            except Exception as error:
                print('Connection with Arduino FAILED: ' + str(error))
                
    def disconnect(self):
        if self.connection is not None:
                self.connection.close()
                print('Successfully disconnect with Arduino')

    def write(self, message):
        try:
            #print('To Arduino:' + message)
            self.connection.write(message.encode())
        except Exception as error:
            print('Write to Arduino FAILED. Error Message: ' + str(error))
            #multithreading will handle the error by stopping the thread, getting reconnection
            raise error 
            # print('Try to reconnect with Arduino')
            # self.disconnect()
            # time.sleep(5)
            # self.connect()

    def read(self):
        try:
            #print("HELLO")
            message = self.connection.readline().strip().decode()
            print("received from Arduino "+message)
            '''counter = 0
            message_list = []
            while (counter < 5):
                message = self.connection.read().decode("utf-8")
                if(message == "|"):
                    counter += 1
                message_list.append(message)
            message = self.connection.read().decode("utf-8")
            message_list.append(message)
            if( message == "-"):
                message = self.connection.read().decode("utf-8")
                message_list.append(message)
            
            message = ''.join(message_list)'''
            if len(message)>0:
               #print("it works " + message)
               return message
            else:
               print("nothing read")
            return None
        except Exception as error:
            print('Read from Arduino FAILED. Error Message: ' + error)
            #multithreading will handle the error by stopping the thread, getting reconnection
            raise error 
            # print('Try to reconnect with Arduino')
            # self.disconnect()
            # time.sleep(5)
            # self.connect()
