import cv2
import find as f
import os.path

last_blue_roi_shape = (0, 0)
last_blue_predicted_number = 0
last_green_roi_shape = (0, 0)
last_green_predicted_number = 0
blue_numbers = []
green_numbers = []

DIR = 'D:\\Boban\Fakultet\Soft\Projekti\softPredef'
file = open(DIR+'\\out.txt','w')
file.write('RA 89/2014 Boban Poznanovic\nfile	sum\n')

DIR = DIR+'\\videos'

video_names=[]
for name in os.listdir(DIR):
    if os.path.isfile(os.path.join(DIR, name)):
        video_names.append(os.path.join(DIR, name))
#
# # print(video_names)

for vid_num in range(0, len(video_names)):
    cap = cv2.VideoCapture(video_names[vid_num])

    blue_element = [last_blue_predicted_number, last_blue_roi_shape]
    blue_numbers.append(blue_element)

    green_element = [last_green_predicted_number, last_green_roi_shape]
    green_numbers.append(green_element)

    #cap = cv2.VideoCapture('videos/video-0.avi')

    ret, frame = cap.read()
    [(b_x1, b_y1), (b_x2, b_y2)] = f.find_blue(frame) #Linija za sabiranje
    [(g_x1, g_y1), (g_x2, g_y2)] = f.find_green(frame) #Linija za oduzimanje
    line_points = []
    line_points.append([(b_x1, b_y1), (b_x2, b_y2)])
    line_points.append([(g_x1, g_y1), (g_x2, g_y2)])

    suma = 0

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        cv2.line(frame, (b_x1, b_y1), (b_x2, b_y2), (255, 255, 255), 2)
        cv2.line(frame, (g_x1, g_y1), (g_x2, g_y2), (255, 255, 255), 2)

        frame, suma, retVal = f.find_numbers_clean(frame, line_points, suma, blue_numbers, green_numbers)

        # last_blue_predicted_number = retVal[0]
        # last_blue_roi_shape = retVal[1]
        # last_green_predicted_number = retVal[2]
        # last_green_roi_shape = retVal[3]
        blue_numbers = retVal[0]
        green_numbers = retVal[1]

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, str(suma), (450, 450), font, 2, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    print('Finalna suma')
    print(suma)
    print(video_names[vid_num])
    print('Suma: ' + str(suma) + '\n')
    file.write('video-' + str(vid_num) + '.avi\t ' + str(suma) + '\n')

    blue_numbers = []
    blue_element = [last_blue_predicted_number, last_blue_roi_shape]
    blue_numbers.append(blue_element)

    green_numbers = []
    green_element = [last_green_predicted_number, last_green_roi_shape]
    green_numbers.append(green_element)

cap.release()
cv2.destroyAllWindows()


