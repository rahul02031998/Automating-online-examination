import cv2
from eye_tracker import eyeDetector
from mouth_opening_detector import mouthCommandR

def runAll():
    while(True):
        eyeDetector()
        mouthCommandR()
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break
