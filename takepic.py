import time
from picamera import PiCamera
from picamera.array import PiRGBArray
import base64
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
    except Exception as error:
        print('Taking picture failed: ' + str(error))
    
    return True

a = _take_pic()
print(a)


                

