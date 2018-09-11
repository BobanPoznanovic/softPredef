import cv2
import sys
import find as f
import numpy as np
import neural_network as nn
import os.path
import matplotlib.pyplot as plt

tracker = cv2.TrackerMedianFlow_create()
counter = 0
colors = []
limit = 4
kernel = np.ones((3, 3), np.uint8)
rectangle_distance_limit = 2
suma = 0
last_blue_roi_shape = (0,0)
last_blue_predicted_number = 0
last_green_roi_shape = (0,0)
last_green_predicted_number = 0

DIR = 'D:\\Boban\Fakultet\Soft\Projekti\softPredef'
file = open(DIR+'\\out.txt','w')
file.write('RA 89/2014 Boban Poznanovic\nfile	sum\n')

DIR = DIR+'\\videos'

video_names=[]
for name in os.listdir(DIR):
    if os.path.isfile(os.path.join(DIR, name)):
        video_names.append(os.path.join(DIR, name))

for vid_num in range(0, len(video_names)):
    cap = cv2.VideoCapture(video_names[vid_num])

    #cap = cv2.VideoCapture("videos/video-9.avi")
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

        if (counter % 10) == 0:
            # Analyse current trackers

            # Create new tracker
            multiTracker = cv2.MultiTracker_create()
            bboxes = f.first_frame_rectangles(frame)
            for bbox in bboxes:
                multiTracker.add(cv2.TrackerMedianFlow_create(), frame, bbox)
                colors.append((255, 0, 0))

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
                            output_shape, input_image = f.pre_detect(frame, kernel, xmin, xmax, ymin, ymax)
                            number = nn.predict(input_image)
                            if number == last_blue_predicted_number:
                                if output_shape[0] == last_blue_roi_shape[0] and output_shape[1] == last_blue_roi_shape[1]:
                                    print()
                                else:
                                    suma = suma + number
                                    last_blue_roi_shape = output_shape
                                    last_blue_predicted_number = number
                            else:
                                suma = suma + number
                                last_roi_shape = output_shape
                                last_predicted_number = number
                            print(number)

                else:
                    if f.above_line(GreenA, GreenB, newbox):
                        if f.rectangle_close_to_line(GreenA, GreenB, newbox, rectangle_distance_limit):
                            image = frame.copy()
                            A, B, C, D = f.rect_to_points(newbox)
                            xmin, xmax, ymin, ymax = f.get_offset_values(A, B, C, D)
                            output_shape, input_image = f.pre_detect(frame, kernel, xmin, xmax, ymin, ymax)
                            number = nn.predict(input_image)
                            if number == last_green_predicted_number:
                                if output_shape[0] == last_green_roi_shape[0] and output_shape[1] == last_green_roi_shape[1]:
                                    print()
                                else:
                                    suma = suma - number
                                    last_green_roi_shape = output_shape
                                    last_green_predicted_number = number
                            else:
                                suma = suma - number
                                last_green_roi_shape = output_shape
                                last_green_predicted_number = number
                            print(number)

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
    print(video_names[vid_num])
    print('Suma: ' + str(suma) + '\n')
    file.write('video-' + str(vid_num) + '.avi\t ' + str(suma) + '\n')
cap.release()
cv2.destroyAllWindows()