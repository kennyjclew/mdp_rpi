from Arduino_rachael import Arduino
#from Android_kenny import Android
import time

arduino = Arduino()
arduino.connect()
apple = 'A|'
arduino.write(apple)
arduino.read()

#android = Android()
#android.connect()
#android.read()
