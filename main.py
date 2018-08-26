import cv2
import numpy as np
from keras.models import load_model
import neural_network as nn
import matplotlib.pyplot as plt
import random
import os.path


def find_blue(input_frame):
    frame_for_blue = input_frame.copy()
    frame_for_blue[:, :, 1] = 0

    gray = cv2.cvtColor(frame_for_blue, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(th, 1, np.pi / 180, 100, minLineLength, maxLineGap)

    # cv2.imshow('blue', t)

    x1 = min(lines[:, 0, 0])
    y1 = max(lines[:, 0, 1])
    x2 = max(lines[:, 0, 2])
    y2 = min(lines[:, 0, 3])
    return [(x1, y1), (x2, y2)]


def find_green(input_frame):
    frame_for_green = input_frame.copy()
    frame_for_green[:, :, 0] = 0

    gray = cv2.cvtColor(frame_for_green, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(th, 1, np.pi / 180, 100, minLineLength, maxLineGap)

    # cv2.imshow('blue', t)

    x1 = min(lines[:, 0, 0])
    y1 = max(lines[:, 0, 1])
    x2 = max(lines[:, 0, 2])
    y2 = min(lines[:, 0, 3])
    return [(x1, y1), (x2, y2)]

def point_to_line_distance(A,B,C):
    # A and B are part of the line, C belongs to the rectangle
    # Result = I/B
    # I = abs[(y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x]
    # Br = sqrt[(y2-y1)^2 + (x2-x1)^2]
    (x1, y1) = A
    (x2, y2) = B
    (x0, y0) = C

    I = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
    Br = np.sqrt(np.square(y2-y1) + np.square(x2-x1))

    return I/Br

def point_to_point_distance(A,B):
    # A belongs to the line, B is point of the rectangle
    (x1, y1) = A
    (x2, y2) = B

    return np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))

def point_close_to_line(A, B, C, limit):
    # Distance between the line and point of rectangle
    # Limit is the distance
    d = point_to_line_distance(A, B, C)

    if d < limit:
        # Distance from line's end-points is not greater than 20
        dA = point_to_point_distance(A, C)
        dB = point_to_point_distance(B, C)
        AtoB = point_to_point_distance(A, B)

        dEnd = min(dA, dB)

        if dEnd < limit:
            return True
        else:
            if dA < AtoB and dB < AtoB:
                return  True
            else:
                return False
    else:
        return False


def rectangle_close_to_line(X1,X2,rect, limit):
    # X1 and X2 are ends of line
    (x, y, w, h) = rect
    A = (x, y)
    B = (x + w, y)
    C = (x, y + h)
    D = (x + w, y + h)

    dA = point_close_to_line(X1, X2, A, limit)
    dB = point_close_to_line(X1, X2, B, limit)
    dC = point_close_to_line(X1, X2, C, limit)
    dD = point_close_to_line(X1, X2, D, limit)

    if dA or dB or dC or dD:
        return True
    else:
        return False

def closest_point_to_line(X1, X2, rect):
    # Returns closest point to the line
    (x, y, w, h) = rect
    A = (x, y)
    B = (x + w, y)
    C = (x, y + h)
    D = (x + w, y + h)

    dA = point_to_line_distance(X1, X2, A)
    dB = point_to_line_distance(X1, X2, B)
    dC = point_to_line_distance(X1, X2, C)
    dD = point_to_line_distance(X1, X2, D)

    minDist = min(dA, dB, dC, dD)

    if dA == minDist:
        return A
    elif dB == minDist:
        return B
    elif dC == minDist:
        return  C
    else:
        return D


def above_line(X1, X2, rect):
    # X1 and X2 are ends of the line
    (x1, y1) = X1
    (x2, y2) = X2

    closestPoint = closest_point_to_line(X1, X2, rect)
    (x0, y0) = closestPoint

    k = (y1-y2)/(x1-x2)
    n = (y1*(x1-x2)-x1*(y1-y2))/(x1-x2)

    rightSide = k*x0 + n

    if y0 > rightSide:
        return  True
    else:
        return False


def find_numbers(frame, lines_points, suma):
    suma = suma

    frame_for_numbers = frame.copy()

    gray = cv2.cvtColor(frame_for_numbers, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    th_cont_img = th.copy()

    img, contours, hierarchy = cv2.findContours(th_cont_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]

    number_contours = []
    number_rect = []

    for contour in contours:
        center, size, angle = cv2.minAreaRect(contour)
        (x, y, w, h) = cv2.boundingRect(contour)
        width, height = size
        if width > 1 and width < 30 and height > 1 and height < 30:
            if width > 9 or height > 9:
                number_contours.append(contour)
                number_rect.append((x, y, w, h))

    img = frame.copy()
    cv2.drawContours(img, number_contours, -1, (0, 0, 255), 1)

    BlueA = lines_points[0][0]
    BlueB = lines_points[0][1]
    GreenA = lines_points[1][0]
    GreenB = lines_points[1][1]

    minDistBlue = 0
    minDistGreen = 0

    for rect in number_rect:
        (x, y, w, h) = rect
        A = (x, y)
        B = (x + w, y)
        C = (x, y + h)
        D = (x + w, y + h)

        limit = 1

        # Check if close to Blue line
        closeToBlueLine = rectangle_close_to_line(BlueA, BlueB, rect, limit)
        closeToGreenLine = rectangle_close_to_line(GreenA, GreenB, rect, limit)

        if closeToBlueLine or closeToGreenLine:
            # Rectangle is close to one line:
            # Udaljenost temena A(x,y) od linija
            dA_b = point_to_line_distance(BlueA, BlueB, A)
            dA_g = point_to_line_distance(GreenA, GreenB, A)

            # Udaljenost temena B(x+w, y) od linija
            dB_b = point_to_line_distance(BlueA, BlueB, B)
            dB_g = point_to_line_distance(GreenA, GreenB, B)

            # Udaljenost temana C(x, y+h) od linija
            dC_b = point_to_line_distance(BlueA, BlueB, C)
            dC_g = point_to_line_distance(GreenA, GreenB, C)

            # Udaljeost temena D(x+w, y+h) od linija
            dD_b = point_to_line_distance(BlueA, BlueB, D)
            dD_g = point_to_line_distance(GreenA, GreenB, D)

            minDistBlue = min(dA_b, dB_b, dC_b, dD_b)
            minDistGreen = min(dA_g, dB_g, dC_g, dD_g)

            if minDistBlue < minDistGreen:
                # Paint rectangle in blue
                if above_line(BlueA, BlueB, rect):
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            else:
                if above_line(GreenA, GreenB, rect):
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Close to lines

            roi = frame[y: y+h, x: x+w]

            #input = cv2.resize(roi, (28,28), interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            ret, th = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

            th = cv2.resize(th, (28,28), interpolation=cv2.INTER_LINEAR)

            # plt.imshow(th)
            # plt.show()

            #number = nn.predict(th)
            #print(number)
            # Detected number
            number = random.randint(0,9)

            if minDistBlue < minDistGreen:
                suma = suma + number
            else:
                suma = suma - number



        else:
            # Rectangle is not close any line:
            aboveBlue = above_line(BlueA, BlueB, rect)
            aboveGreen = above_line(GreenA, GreenB, rect)
            if not aboveGreen or not aboveBlue:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Take an number

    return img, suma




DIR = 'D:\\Boban\Fakultet\Soft\Projekti\softPredef'
file = open(DIR+'\\out.txt','w')
file.write('RA 89/2014 Boban Poznanovic\nfile	sum\n')

DIR = DIR+'\\videos'

video_names=[]
for name in os.listdir(DIR):
    if os.path.isfile(os.path.join(DIR, name)):
        video_names.append(os.path.join(DIR, name))

print(video_names)

for vid in range(0, len(video_names)):
    cap = cv2.VideoCapture(video_names[vid])

    ret, frame = cap.read()
    [(b_x1, b_y1), (b_x2, b_y2)] = find_blue(frame)  # Linija za sabiranje
    [(g_x1, g_y1), (g_x2, g_y2)] = find_green(frame)  # Linija za oduzimanje
    lines_points = []
    lines_points.append([(b_x1, b_y1), (b_x2, b_y2)])
    lines_points.append([(g_x1, g_y1), (g_x2, g_y2)])

    # model = load_model('mnist_model.h5')

    suma = 0

    while (cap.isOpened()):
        ret, frame = cap.read()

        frame, suma = find_numbers(frame, lines_points, suma)

        cv2.line(frame, (b_x1, b_y1), (b_x2, b_y2), (0, 0, 255), 2)
        cv2.line(frame, (g_x1, g_y1), (g_x2, g_y2), (0, 0, 255), 2)

        # cv2.imshow('frame', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    print(video_names[vid])
    print('Suma: ' + str(suma) + '\n')
    file.write('video-' + str(vid) + '.avi\t ' + str(suma) + '\n')

cap.release()
cv2.destroyAllWindows()
