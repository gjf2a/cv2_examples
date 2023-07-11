# From perplexity.ai: "write a python opencv program doing simple image processing on a live feed"

import cv2
import numpy as np

from morph_contour_demo import Timer

cap = cv2.VideoCapture(1)
timer = Timer()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Contour material from https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0,0,255), 3)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    timer.inc()

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
fps = timer.elapsed()
cap.release()
cv2.destroyAllWindows()

print(fps)
