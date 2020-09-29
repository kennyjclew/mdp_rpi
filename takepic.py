

def _take_pic():
    try:
        start_time = datetime.now()

        # initialize the camera and grab a reference to the raw camera capture
        camera = PiCamera(resolution=(1920, 1080))  # '1920x1080'
        rawCapture = PiRGBArray(camera)
        
        # allow the camera to warmup
        time.sleep(0.1)
        
        # grab an image from the camera
        camera.capture(rawCapture, format='bgr')
        image = rawCapture.array
        camera.close()

        print('Time taken to take picture: ' + str(datetime.now() - start_time) + 'seconds')
        
        # to gather training images
        # os.system("raspistill -o images/test"+
        # str(start_time.strftime("%d%m%H%M%S"))+".png -w 1920 -h 1080 -q 100")
    
    except Exception as error:
        print('Taking picture failed: ' + str(error))
    
    return image

a = _take_pic()
print(a)


                

