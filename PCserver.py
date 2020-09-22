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
            print("trying to connect...")
            self.conn, self.addr = self.s.accept() #gets client socket object conn
            if (self.conn and self.addr) is not None:
                print("connected!")
                connected = True


    #close client socket
    def disconnect(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None
            self.addr = None
            print("client socket closed")

    #close both client and server socket - unnecessary?
    def disconnect_both(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None
            self.addr = None
            print("client socket closed")
        if self.s is not None:
            self.s.close()
            self.s = None
            print("server socket closed")

    #receive data from client
    def read(self):
        data = self.conn.recv(1024).decode('utf-8')
        print("server received data " + data)
        if len(data) > 0:
            return data
        else:
            return None

    #send data to client
    def write(self, data):
        self.conn.send(data.encode('utf-8'))
        print("server sending data " + data)











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
