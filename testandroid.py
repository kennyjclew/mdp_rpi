from Android_kenny import Android
import os

os.system("sudo hciconfig hci0 piscan")

android = Android()
android.connect()
android.read()
