# This will be an object oriented version of the virtual game

import cv2 
import numpy as np

class FaceFinder:
    """This is going to use haarcascade as filter to detect the largest face"""

    def __init__(self):
        print('facefinder initialized')
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def find_face(self,frame):
        """ Returns face center (x,y), draws Rect on frame """

        # Convert to grayscale

        gray = cv2.cvtColor(frame,cv2.BGR2GRAY)
        faces = self.face_cascade.deleteMultiScale(gray,minNeighbors = 9)

        # Draw rectangle

        if faces is None:
            
            return None

        bx, by, bw, bh = 0 

        for (x, y, w, h) in faces:

            if w>bw: # If current face is bigger than biggest found so far

                bw, by, bw, bh = x, y, w, h
          
        cv2.rectangle(img, (bx, by), (bx+bw, by+bh), (0, 255, 255), 3)

        return (bx+bw/2),(by+bh/2)
#--------------------------------------------------------------------
# main
ff = FaceFinder()
print('Starting O.O. virtual3D')