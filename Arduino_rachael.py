import serial
import time

class Arduino:
    #information about Arduino
    def init(self):
        # self.serial_port = "/dev/ttyUSB0"
        self.serial_port = '/dev/ttyACM0'
        self.baud_rate = 115200
        self.connection = None

    def connect(self):
        print('Connecting to Arduino...')
        try:
            self.connection = serial.Serial('/dev/ttyACM0', 115200)
            if self.connection is not None: # It is connected to at least 1 of the Arduino
                print('Connection with Arduino ' + self.connection.name + ' SUCCESSFUL')
            time.sleep(2)
        except Exception as error:
            print('Connection with Arduino FAILED')
            print('Try to reconnect with Arduino')
            self.disconnect()
            time.sleep(5)
            self.connect()

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
            print('Try to reconnect with Arduino')
            self.disconnect()
            time.sleep(5)
            self.connect()

    def read(self):
        try:
            print("HELLO")
            message = self.connection.readline().strip().decode()
            print(message)
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
               print("it works " + message)
               return message
            else:
               print("nothing read")
            return None
        except Exception as error:
            print('Read from Arduino FAILED. Error Message: ' + error)
            print('Try to reconnect with Arduino')
            self.disconnect()
            time.sleep(5)
            self.connect()
