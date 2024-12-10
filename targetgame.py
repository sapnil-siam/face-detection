import cv2
import numpy as np

class FaceFinder:
    """Initializes a face cascade, detects faces, finds the largest, and draws face rectangles."""

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def find_face(self, frame):
        """Detects faces and draws rectangles around the largest one. Returns the center coordinates."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=9)
        if len(faces) == 0:
            return None

        bx = by = bw = bh = 0
        for (x, y, w, h) in faces:
            if w > bw:  # Select the largest face
                bx, by, bw, bh = x, y, w, h

        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 255), 3)
        return (bx + bw // 2, by + bh // 2)

class Stage:
    """Manages the display and drawing of the game."""

    def __init__(self):
        self.disp_h = 1080
        self.disp_w = 1920
        self.cam_h = 720
        self.cam_w = 1280
        self.save_x = 960

    def draw_target_xy(self, img, pos, size):
        """Draws concentric target circles at the specified position."""
        cv2.circle(img, pos, size, (0, 0, 255), -1)
        cv2.circle(img, pos, int(size * 0.8), (255, 255, 255), -1)
        cv2.circle(img, pos, int(size * 0.6), (0, 0, 255), -1)
        cv2.circle(img, pos, int(size * 0.4), (255, 255, 255), -1)
        cv2.circle(img, pos, int(size * 0.2), (0, 0, 255), -1)

    def update(self, facexy):
        """Updates the stage display based on face position."""
        x, y = facexy
        e = 0.9  # Smoothing constant
        x = e * x + (1 - e) * self.save_x
        self.save_x = x

        img = np.zeros((self.disp_h, self.disp_w, 3), dtype=np.uint8)
        decay = 0.3

        sx = sy = 0
        dx = int((x - self.cam_w / 2) * 2)
        for i in range(1, 7):  # Draw grid
            sx = sx + int((960 - sx) * decay)
            sy = sy + int((540 - sy) * decay)
            dx = int(dx * decay)
            cv2.rectangle(img, (sx + dx, sy), (self.disp_w - sx + dx, self.disp_h - sy), (255, 255, 255), 1)

        # Draw targets
        ball_positions = [
            (600 + int((x - self.cam_w / 2) * 2 * 0.6), 540, 35),
            (1000 + int((x - self.cam_w / 2) * 2 * 0.2), 440, 25),
            (1100 + int((x - self.cam_w / 2) * 2 * 0.9), 650, 50)
        ]

        for ballx, bally, size in ball_positions:
            self.draw_target_xy(img, (ballx, bally), size)

        cv2.imshow("Siam's Game", img)

# Main application
ff = FaceFinder()
stage = Stage()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('Cannot open camera. Ensure the webcam is available and not used by another app.')
    exit()

moved = False
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame. Exiting...")
        break

    facexy = ff.find_face(frame)
    frame_small = cv2.resize(frame, (frame.shape[1] // 4, frame.shape[0] // 4), interpolation=cv2.INTER_LINEAR)
    cv2.imshow('press q to quit', frame_small)
    if not moved:
        cv2.moveWindow('press q to quit', 1080, 0)
        moved = True

    if facexy is not None:
        stage.update(facexy)

    # Stop if 'q' key is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()
