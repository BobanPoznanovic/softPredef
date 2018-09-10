import cv2
import sys
import find as f
import numpy as np
import neural_network as nn
import matplotlib.pyplot as plt

tracker = cv2.TrackerMedianFlow_create()
counter = 0
colors = []
limit = 2
kernel = np.ones((3, 3), np.uint8)
rectangle_distance_limit = 14
suma = 0

cap = cv2.VideoCapture("videos/video-1.avi")
success, frame = cap.read()

if not success:
    print("Failed to read video")
    sys.exit(1)

# Find end of Blue and Green line
[BlueA, BlueB] = f.find_blue(frame) #Linija za sabiranje
[GreenA, GreenB] = f.find_green(frame) #Linija za oduzimanje
bboxes = f.first_frame_rectangles(frame)

multiTracker = cv2.MultiTracker_create()

for bbox in bboxes:
    multiTracker.add(cv2.TrackerMedianFlow_create(), frame, bbox)
    colors.append((255, 0, 0))

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # if (counter % 10) == 0:
    #     # Analyse current trackers
    #
    #     # Create new tracker
    #     multiTracker = cv2.MultiTracker_create()
    #     bboxes = f.first_frame_rectangles(frame)
    #     for bbox in bboxes:
    #         multiTracker.add(cv2.TrackerMedianFlow_create(), frame, bbox)
    #         colors.append((255, 0, 0))

    success, boxes = multiTracker.update(frame)

    for i, newbox in enumerate(boxes):
        blue_is_closer, dist = f.rectangle_to_closest_line_distance(BlueA, BlueB, GreenA, GreenB, newbox)

        if dist < limit:
            if blue_is_closer:
                if f.above_line(BlueA, BlueB, newbox):
                    if f.rectangle_close_to_line(BlueA, BlueB, newbox, rectangle_distance_limit):
                        # frame, newbox(~rect), kernel
                        image = frame.copy()
                        A, B, C, D = f.rect_to_points(newbox)
                        xmin, xmax, ymin, ymax = f.get_offset_values(A, B, C, D)
                        input_image = f.pre_detect(frame, kernel, xmin, xmax, ymin, ymax)
                        number = nn.predict(input_image)
                        print(number)
                        suma = suma + number
            else:
                if f.above_line(GreenA, GreenB, newbox):
                    if f.rectangle_close_to_line(GreenA, GreenB, newbox, rectangle_distance_limit):
                        image = frame.copy()
                        A, B, C, D = f.rect_to_points(newbox)
                        xmin, xmax, ymin, ymax = f.get_offset_values(A, B, C, D)
                        input_image = f.pre_detect(frame, kernel, xmin, xmax, ymin, ymax)
                        number = nn.predict(input_image)
                        print(number)
                        suma = suma - number

        # Just for overview
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
        #cv2.putText(frame, str(dist), p2, cv2.FONT_ITALIC, 1, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('MultiTracker', frame)

    counter = counter + 1

    if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
        break

    # print("Temp suma: " + str(suma))

print("Suma: " + str(suma))