from PCserver import PCserver
import time
from picamera import PiCamera
from picamera.array import PiRGBArray
import base64
#from PCclient import PCclient

HOST = '192.168.21.21'  # Standard loopback interface address (localhost)
PORT = 65432        # Port to listen on (non-privileged ports are > 1023)

server = PCserver(HOST, PORT)
# client = PCclient(HOST, PORT)

server.connect()
# client.connect()
server.read()
server.write("hello world!")

def _take_pic():
    try:
       # start_time = datetime.now()

        # initialize the camera and grab a reference to the raw camera capture
        camera = PiCamera(resolution=(1920, 1080))  # '1920x1080'
        camera.hflip = True 
        rawCapture = PiRGBArray(camera)
        # allow the camera to warmup
        time.sleep(0.1)
        
        # grab an image from the camera
        camera.capture(rawCapture, format='bgr')
        camera.capture('foo.jpg')
        with open("foo.jpg", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        print(type(encoded_string))
        text_file = open("sample.txt", "w")
        n = text_file.write(encoded_string.decode())
        text_file.close()
       # image = rawCapture.array
        camera.close()
        return encoded_string
    except Exception as error:
        print('Taking picture failed: ' + str(error))
    
    return True

a = _take_pic()


server.write(encoded_string.decode());
server.write("hello world")
#server.read()
# client.receive()
print("sent all already")
server.disconnect()
server.disconnect_both()
