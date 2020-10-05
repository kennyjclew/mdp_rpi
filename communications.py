HOST_IP = '192.168.21.21'
PORT = 65431

MSG_SEPARATOR = '|'.encode()
NEWLINE = '\n'.encode()

ANDROID_HEADER = 'ANDROID'.encode()
ARDUINO_HEADER = 'ARDUINO'.encode()
PC_HEADER = 'PC'.encode()

class RobotStatus:
    IDLE = 'idle'.encode()
    EXPLORING = 'exploring'.encode()
    FASTEST_PATH = 'fastestpath'.encode()

class AndroidToArduino:
    MOVE_FORWARD = 'W|'.encode()
    MOVE_BACK = 'S|'.encode()
    TURN_LEFT = 'A|'.encode()
    TURN_RIGHT = 'D|'.encode()
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
    START_FASTEST_PATH = 'F|'.encode()

    ANDTORPI_MESSAGES = [START_EXPLORATION, START_FASTEST_PATH]

# class AndroidToPC:
#     UPDATE_ARENA = 'UA|'.encode()

#Arduino messages (about distances) will be passed directly to algo 

class PCToAndroid:
    #pc passes mapstring immediately after passing 'M'
        #e.g. pc passes 'M00101...'
    MAP_STRING = 'M'.encode()

    # MOVE_FORWARD = 'W'.encode()[0]
    # TURN_LEFT = 'A'.encode()[0]
    # TURN_RIGHT = 'D'.encode()[0]
    # CALIBRATING_CORNER = 'L'.encode()[0]
    # SENSE_ALL = 'Z'.encode()[0]
    # ALIGN_RIGHT = 'B'.encode()[0]
    # ALIGN_FRONT = 'V'.encode()[0]

class PCToRPi:
    TAKE_PICTURE = 'TP'
    EXPLORATION_DONE = 'ED'.encode()


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
    PICTURE_TAKEN = 'PT'.encode()
    DONE_IMG_REC = 'I'.encode()
