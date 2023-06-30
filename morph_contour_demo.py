# From perplexity.ai: "write a python opencv program doing simple image processing on a live feed"

import sys
import cv2
import numpy as np


def morph_contour_loop(video_port, kernel_side):
    kernel_size = (kernel_side, kernel_side)
    cap = cv2.VideoCapture(video_port)
    while True:
        ret, frame = cap.read()
        contours, hierarchy = find_contours(frame, kernel_size)
        filtered_contours = []
        for contour in contours:
            height_filter = contour[:, :, 1] > 200
            result = np.zeros_like(contour)
            result[height_filter] = contour[height_filter]
            result = result[np.any(result != 0, axis=2)]
            if len(result) > 0:
                filtered_contours.append(result)

        cv2.drawContours(frame, filtered_contours, -1, (0, 0, 255), 3)
        #cv2.drawContours(frame, contours, -1, (0, 0, 255), 3)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def find_contours(frame, kernel_size):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # From https://www.scaler.com/topics/contour-analysis-opencv/
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    filtered = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # Contour material from https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
    ret, thresh = cv2.threshold(filtered, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: morph_contour_demo.py video_port kernel_side")
    else:
        morph_contour_loop(int(sys.argv[1]), int(sys.argv[2]))

