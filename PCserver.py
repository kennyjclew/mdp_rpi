"""
refs: 
    https://www.binarytides.com/python-socket-programming-tutorial/
    https://realpython.com/python-sockets/

"""

import socket

# HOST_IP = ''
# PORT = 8888 #port to listen on. to change

class PCserver:
    #setup server socket, listen for client connections
    def __init__(self, HOST_IP, PORT):
        self.conn = None
        self.addr = None
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #AF_INET is internet address family for IPv4, SOCK_STREAM is socket type for TCP (protocol)

        #not sure if need to setsockopt yet. previous execution may result in "Address already in use" error
        #s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        #bind socket to specific network interface and port number. IPv4 expects a 2-tuple
        self.s.bind((HOST_IP, PORT))
        
        self.s.listen(1) #arg: number of unaccepted connections kept waiting (queue length). if not specified, default backlog is chosen

    #connect client(PC) to server(RPi)
    def connect(self):
        connected = False
        while not connected:
            try:
                print("trying to connect to pc...")
                self.conn, self.addr = self.s.accept() #gets client socket object conn
                if (self.conn and self.addr) is not None:
                    print("connected to pc!")
                    connected = True
            except Exception as error:
                print("connection to pc failed: " + str(error))

                if self.conn is not None:
                    print("closing pc client socket...")
                    self.conn.close()
                    self.conn = None
                    self.addr = None



    #close client socket
    def disconnect(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None
            self.addr = None
            print("pc client socket closed")

    #close both client and server socket - unnecessary?
    def disconnect_both(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None
            self.addr = None
            print("pc client socket closed")
        if self.s is not None:
            self.s.close()
            self.s = None
            print("rpi's pcserver socket closed")

    def checkconnections(self):
        if (self.conn is None) or (self.s is None):
            return False
        else:
            return True

    #receive data from client
    def read(self):
        try:
            data = self.conn.recv(1024).decode('utf-8')
            print("rpi received data from pc:  " + data)
            if len(data) > 0:
                return data
            else:
                return None
        except Exception as error:
            print("PC read failed: " + str(error))
            #multithreading will handle the error by stopping the thread, getting reconnection
            raise error 

    #send data to client (pc)
    def write(self, data):
        try:
            self.conn.send((data+'\n').encode('utf-8'))
            print("rpi forwarding data to pc: " + data)
        except Exception as error:
            print("PC write failed: " + str(error))
            #multithreading will handle the error by stopping the thread, getting reconnection
            raise error 










# HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
# PORT = 65432        # Port to listen on (non-privileged ports are > 1023)

# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# s.bind((HOST, PORT))
# s.listen()
# conn, addr = s.accept()
# while True:
#     print('Connected by', addr)
#     while True:
#         data = conn.recv(1024)
#         if not data:
#             break
#         conn.sendall(data)
