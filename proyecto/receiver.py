import zmq
import cv2 as cv

from mycodec import decode

import sys


port = 5555
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind(f"tcp://*:{port}")

def getSize(input):
    # Tamaño del mensaje en megabits
    size = round(sys.getsizeof(input)*7.629395e-6,2)
    return size

while True:
    #i=i+1
    message = socket.recv()
    s = getSize(message)
    print(s)
    frame = decode(message)
    cv.imshow("Torres del paine", frame)
    cv.waitKey(10)
    socket.send(b"ready")
