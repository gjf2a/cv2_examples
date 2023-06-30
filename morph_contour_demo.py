import sys
import cv2
import numpy as np


def morph_contour_loop(video_port, kernel_side):
    kernel_size = (kernel_side, kernel_side)
    cap = cv2.VideoCapture(video_port)
    while True:
        ret, frame = cap.read()
        contours, hierarchy = find_contours(frame, kernel_size)

        cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
        #cv2.drawContours(frame, purged_behind(contours), -1, (0, 0, 255), 3)
        cv2.drawContours(frame, close_contour(contours, cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), -1, (0, 0, 255), 3)
        #cv2.drawContours(frame, close_contour(purged_behind(contours), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), -1, (255, 0, 0), 3)

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
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contours, hierarchy


def purged_behind(contours):
    contours = sorted(contours, key=lambda c: -np.max(c[:,:,1]))
    purged = set()
    for i in range(len(contours)):
        min_x_front, max_x_front = contour_x_bounds(contours[i])
        for j in range(i + 1, len(contours)):
            min_x_back, max_x_back = contour_x_bounds(contours[j])
            if min_x_back >= min_x_front and max_x_back <= max_x_front:
                purged.add(j)
    return [contours[i] for i in range(len(contours)) if i not in purged]


def close_contour(contours, height):
    best_xs = {}
    for contour in contours:
        for point in contour:
            if point[0][1] < height - 1 and (point[0][0] not in best_xs or point[0][1] > best_xs[point[0][0]]):
                best_xs[point[0][0]] = point[0][1]
    close_contour = np.empty((len(best_xs), 1, 2), dtype=contours[0].dtype)
    for i, (x, y) in enumerate(best_xs.items()):
        close_contour[i] = np.array([[x, y]])
    return close_contour


def farthest_x_y(contours):
    pass


def contour_x_bounds(contour):
    return np.min(contour[:,:,0]), np.max(contour[:,:,0])


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: morph_contour_demo.py video_port kernel_side")
    else:
        morph_contour_loop(int(sys.argv[1]), int(sys.argv[2]))

