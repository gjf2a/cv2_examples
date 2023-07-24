# Derived in part from https://www.perplexity.ai/search/4bb7f7b6-e404-462b-9ed4-5c53dd016b28?s=c

import cv2
import numpy as np

from morph_contour_demo import Timer

cap = cv2.VideoCapture(1)
timer = Timer()

while True:
    # Capture frame-by-frame
    ret, img = cap.read()

    # Convert the frame to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds of the red color in HSV color space
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # Create a binary mask for the red color using cv2.inRange()
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    red_filtered = cv2.bitwise_and(img, img, mask=mask)

    # Display the resulting frame
    cv2.imshow('original', img)
    cv2.imshow('filtered', red_filtered)
    timer.inc()

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

fps = timer.elapsed()
# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
print(fps)
