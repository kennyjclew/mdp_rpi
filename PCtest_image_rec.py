from PCserver import PCserver
import time
#from PCclient import PCclient

HOST = '192.168.21.21'  # Standard loopback interface address (localhost)
PORT = 65432        # Port to listen on (non-privileged ports are > 1023)

server = PCserver(HOST, PORT)
# client = PCclient(HOST, PORT)

server.connect()
# client.connect()
server.read()
server.write("hello world!")
# time.sleep(2)
# print("hello before reading 3 times")
# msg = server.read()
# #pc either sends to android (only mapstring) or rpi
# msg_list = msg.splitlines()
# for msg in msg_list:
#   if msg is None:
#     continue
#   elif msg[0] == 1: #PC must send MAPSTRING with MAP_STRING as a header
#     msgqueue.put([ANDROID_HEADER, msg])
#   elif msg == "BOOHOO":
#     #KIV: RPI TAKE PICTURE
#     print("pc tells rpi to take picture")
#   elif msg == "HAHAH":
#     #KIV: RPI DO SOMETHING. display all images recognised?
#     print("pc tells rpi that exploration done")
#   else:
#     print("this is message" + msg)
#     print("msg from PC forwarding to arduino")

#server.read()
#server.read()
#server.read()
#server.read()
print("bye bye read")
time.sleep(2)
print('before write')
server.write("E|")
print('hello')
time.sleep(2)
#server.read()
# client.receive()
server.disconnect()
server.disconnect_both()
