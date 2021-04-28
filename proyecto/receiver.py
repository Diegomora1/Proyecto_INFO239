import zmq
import cv2 as cv

from mycodec import decode


port = 5555
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind(f"tcp://*:{port}")

i=0

while True:
    #i=i+1
    message = socket.recv()
    frame = decode(message)
    cv.imshow("Torres del paine", frame)
    cv.waitKey(10)
    socket.send(b"ready")
