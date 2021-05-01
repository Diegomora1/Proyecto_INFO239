import zmq
import cv2 as cv
import numpy as np
import sys
from mycodec import decode
from time import time

def getSize(input):
    # Tamaño del mensaje en megabits
    size = round(sys.getsizeof(input)*7.629395e-6,2)
    return size

port = 5555
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind(f"tcp://*:{port}")



while True:
    t1 = time()
    message = socket.recv()
    t2 = time()
    s = getSize(message)
    tasa = round(s/(t2-t1),2)
    print("Tasa de transferencia: ",tasa," Mbps")
    frame = decode(message)
    cv.imshow("Torres del paine", frame)
    cv.waitKey(10)
    socket.send(b"ready")

