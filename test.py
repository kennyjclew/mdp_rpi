from Arduino_rachael import Arduino
#from Android_kenny import Android
import time

arduino = Arduino()
arduino.connect()
apple = 'A|'
arduino.write(apple)
arduino.write('W|')
arduino.write("R|")

arduino.read()

arduino.write("W|S|")

#android = Android()
#android.connect()
#android.read()
