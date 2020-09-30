HOST_IP = '192.168.21.21'
PORT = 65431

MSG_SEPARATOR = '|'
NEWLINE = '\n'

ANDROID_HEADER = 'ANDROID'
ARDUINO_HEADER = 'ARDUINO'
PC_HEADER = 'PC'

class RobotStatus:
    IDLE = 'idle'
    EXPLORING = 'exploring'
    FASTEST_PATH = 'fastestpath'

class AndroidToArduino:
    MOVE_FORWARD = 'W|'
    MOVE_BACK = 'S|'
    TURN_LEFT = 'A|'
    TURN_RIGHT = 'D|'
    CALIBRATE_SENSOR = 'C|'

    #for rpi to check if msg is valid. 
    # ANDTOARD_MESSAGES = [
    #     MOVE_FORWARD,
    #     MOVE_BACK,
    #     TURN_LEFT,
    #     TURN_RIGHT,
    #     CALIBRATE_SENSOR,
    # ]

#to send to both arduino and pc
class AndroidToRPi:
    START_EXPLORATION = 'E|'
    START_FASTEST_PATH = 'F|'

    ANDTORPI_MESSAGES = [START_EXPLORATION, START_FASTEST_PATH]

# class AndroidToPC:
#     UPDATE_ARENA = 'UA|'.encode()

#Arduino messages (about distances) will be passed directly to algo 

class PCToAndroid:
    #pc passes mapstring immediately after passing 'M'
        #e.g. pc passes 'M00101...'
    MAP_STRING = 'M'

    # MOVE_FORWARD = 'W'.encode()[0]
    # TURN_LEFT = 'A'.encode()[0]
    # TURN_RIGHT = 'D'.encode()[0]
    # CALIBRATING_CORNER = 'L'.encode()[0]
    # SENSE_ALL = 'Z'.encode()[0]
    # ALIGN_RIGHT = 'B'.encode()[0]
    # ALIGN_FRONT = 'V'.encode()[0]

class PCToRPi:
    TAKE_PICTURE = 'TP'
    EXPLORATION_DONE = 'ED'


# class RPiToAndroid:
#     STATUS_EXPLORING = '{"status":"exploring"}'.encode()
#     STATUS_FASTEST_PATH = '{"status":"fastest path"}'.encode()
#     STATUS_TURNING_LEFT = '{"status":"turning left"}'.encode()
#     STATUS_TURNING_RIGHT = '{"status":"turning right"}'.encode()
#     STATUS_IDLE = '{"status":"idle"}'.encode()
#     STATUS_TAKING_PICTURE = '{"status":"taking picture"}'.encode()
#     STATUS_CALIBRATING_CORNER = '{"status":"calibrating corner"}'.encode()
#     STATUS_SENSE_ALL = '{"status":"sense all"}'.encode()
#     STATUS_MOVING_FORWARD = '{"status":"moving forward"}'.encode()
#     STATUS_ALIGN_RIGHT = '{"status":"align right"}'.encode()
#     STATUS_ALIGN_FRONT = '{"status":"align front"}'.encode()
    
#     MOVE_UP = '{"move":[{"direction":"forward"}]}'.encode()
#     TURN_LEFT = '{"move":[{"direction":"left"}]}'.encode()
#     TURN_RIGHT = '{"move":[{"direction":"right"}]}'.encode()


class RPiToPC:
    PICTURE_TAKEN = 'PT'
    DONE_IMG_REC = 'I'
