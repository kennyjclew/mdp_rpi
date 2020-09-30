#import cv2
import imagezmq
import time
from picamera import PiCamera
from multiprocessing import Process, Value, Queue, Manager
from picamera.array import PiRGBArray

class TestImageZMQ():
    """docstring for ClassName"""
    def __init__(self):
        self.image_process = None
        # if image_processing_server_url is not None:
        self.image_process = Process(target=self._process_pic)
        self.manager = Manager()
        # pictures taken using the PiCamera are placed in this queue
        self.image_queue = self.manager.Queue()

        # self.image_processing_server_url = image_processing_server_url
        self.image_count = Value('i',0)


    def _take_pic(self):
        try:
           # start_time = datetime.now()

            # initialize the camera and grab a reference to the raw camera capture
            camera = PiCamera(resolution=(1920,1080))  # '1920x1080'
            camera.hflip = True 
            rawCapture = PiRGBArray(camera)
            
            # allow the camera to warmup
            time.sleep(0.1)
            
            # grab an image from the camera
            camera.capture(rawCapture, format='bgr')
            image = rawCapture.array
            print("image taken")
            camera.close()

           # print('Time taken to take picture: ' + str(datetime.now() - start_time) + 'seconds')
            
            # to gather training images
            # os.system("raspistill -o images/test"+
            # str(start_time.strftime("%d%m%H%M%S"))+".png -w 1920 -h 1080 -q 100")
        
        except Exception as error:
            print('Taking picture failed: ' + str(error))
        
        return image



    def _process_pic(self, image):
        # initialize the ImageSender object with the socket address of the server
        image_sender = imagezmq.ImageSender(connect_to="tcp://192.168.18.11:5555")
        print("connected")
        image_id_list = []
        while True:
            try:
                if 1==1:
                   # start_time = datetime.now()
                    
                    # image_message =  self.image_queue.get_nowait()
                    # format: 'x,y|x,y|x,y'
                    obstacle_coordinates = "1,2"
                    
                    reply = image_sender.send_image(
                        'image from RPi', 
                        image
                    )
                    reply = reply.decode('utf-8')

                    print(reply)
                    if reply == 'End':
                        break  # stop sending images
                    
                    # example replies
                    # "1|2|3" 3 symbols in order from left to right
                    # "1|-1|3" 2 symbols, 1 on the left, 1 on the right
                    # "1" 1 symbol either on the left, middle or right
                    else:
                        detections = reply.split(MESSAGE_SEPARATOR)
                        obstacle_coordinate_list = obstacle_coordinates.split(MESSAGE_SEPARATOR)

                        for detection, coordinates in zip(detections, obstacle_coordinate_list):
                            
                            if coordinates == '-1,-1':
                                continue  # if no obstacle, skip mapping of symbol id
                            elif detection == '-1':
                                continue  # if no symbol detected, skip mapping of symbol id
                            else:
                                id_string_to_android = '{"image":[' + coordinates + \
                                ',' + detection + ']}'
                                print(id_string_to_android)
                                
                                if detection not in image_id_list:
                                    self.image_count.value += 1
                                    image_id_list.put_nowait(detection)
                                
                                self.to_android_message_queue.put_nowait(
                                    id_string_to_android + NEWLINE
                                )

                   # print('Time taken to process image: ' + \
                    #    str(datetime.now() - start_time) + ' seconds')

            except Exception as error:
                print('Image processing failed: ' + str(error))


a = TestImageZMQ()
b = a._take_pic()
print(b)
a._process_pic(b)

print("ended")






