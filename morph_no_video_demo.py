from morph_contour_demo import Timer, contour_inner_loop
import cv2

cap = cv2.VideoCapture(1)
kernel_size = (9, 9)

timer = Timer()

# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

for i in range(20):
    frame, contours, close_contour, best = contour_inner_loop(cap, kernel_size, 20)   
    timer.inc()
    print(best)
print(timer.elapsed())
cap.release()
