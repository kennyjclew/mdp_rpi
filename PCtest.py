from PCserver import PCserver
#from PCclient import PCclient

HOST = '192.168.21.21'  # Standard loopback interface address (localhost)
PORT = 65431        # Port to listen on (non-privileged ports are > 1023)

server = PCserver(HOST, PORT)
# client = PCclient(HOST, PORT)

server.connect()
# client.connect()
# client.send("hello world!")
server.receive()
server.send("hello world!")
# client.receive()
server.disconnect()
server.disconnect_both()
