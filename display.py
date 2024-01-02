import cv2


class Display():
    def __init__(self, W,H):
        self.W=W
        self.H=H
        self.window_name = 'My Video'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL) 
        cv2.resizeWindow(self.window_name, W, H)
    
    def paint(self,img):
        cv2.imshow(self.window_name, img)