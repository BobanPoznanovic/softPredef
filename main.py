import cv2
import find as f
import os.path

DIR = 'D:\\Boban\Fakultet\Soft\Projekti\softPredef'
file = open(DIR+'\\out.txt','w')
file.write('RA 89/2014 Boban Poznanovic\nfile	sum\n')

DIR = DIR+'\\videos'

# video_names=[]
# for name in os.listdir(DIR):
#     if os.path.isfile(os.path.join(DIR, name)):
#         video_names.append(os.path.join(DIR, name))
#
# # print(video_names)

cap = cv2.VideoCapture('videos/video-0.avi')

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

    frame, suma = f.find_numbers(frame, line_points, suma)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, str(suma), (450, 450), font, 2, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

print('Finalna suma')
print(suma)
cap.release()
cv2.destroyAllWindows()


