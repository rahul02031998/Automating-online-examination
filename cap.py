import cv2

def openCamera():
    return cv2.VideoCapture(0)

def destroy():
    openCamera().release()
    cv2.destroyAllWindows()

def destroyAll():
    cv2.destroyAllWindows()