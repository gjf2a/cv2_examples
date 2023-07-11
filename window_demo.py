# From perplexity.ai: "write a python opencv program doing simple image processing on a live feed"

import cv2

from morph_contour_demo import Timer

cap = cv2.VideoCapture(1)
timer = Timer()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame', gray)
    timer.inc()

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

fps = timer.elapsed()
# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
print(fps)
