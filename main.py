from multithreading import Multithreading 
import time
try:
    processes = Multithreading()
except KeyboardInterrupt:
    print("stopping all threads and disconnecting all devices")
    processes.stop_all()
    print("the end")