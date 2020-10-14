from multithreading import Multithreading 
import time
import os

os.system("sudo hciconfig hci0 piscan")

try:
    processes = Multithreading()
    print("starting multithreading")
    processes.start_all_threads()
except KeyboardInterrupt:
    print("stopping all threads and disconnecting all devices")
    processes.stop_all()
    print("the end")

