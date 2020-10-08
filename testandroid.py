from Android_kenny import Android
import os

os.system("sudo hciconfig hci0 piscan")

android = Android()
android.connect()
android.read()
waypoint = android.read()
if(waypoint[:8] == "waypoint"):
    print(waypoint)
    waypoint_coord = waypoint[10:-1]
    print(waypoint_coord)
    coord_array = waypoint_coord.split(",")
    print(coord_array)
android.read()
