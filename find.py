import cv2
import numpy as np
import neural_network as nn
import matplotlib.pyplot as plt

last_roi_shape = []
last_blue_roi_shape = (0, 0)
last_blue_predicted_number = 0
last_green_roi_shape = (0, 0)
last_green_predicted_number = 0


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

    k = (y1 - y2)/(x1 - x2)
    n = (y1*(x1 - x2) - x1*(y1 - y2))/(x1 - x2)

    a = -k
    b = 1
    c = -n

    I = np.absolute(a*x0 + b*y0 + c)
    Br = np.sqrt(np.square(a) + np.square(b))

    # a1 = y2 - y1
    # b1 = x2 - x1
    #
    # a = a1*x0
    # b = b1*y0
    # c = x2*y1
    # d = y2*x1
    #
    # I = np.absolute((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 + y2*x1)
    # Br = np.sqrt(np.square(y2-y1) + np.square(x2-x1))

    result = I/Br

    return result

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
        # Distance from line's end-points is not greater than limit
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


def rectangle_close_to_line(X1,X2, rect, limit):
    # X1 and X2 are ends of line
    A, B, C, D = rect_to_points(rect)

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
    A, B, C, D = rect_to_points(rect)

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
        return C
    else:
        return D

def rectangle_to_line_distance(X1, X2, rect):
    point = closest_point_to_line(X1, X2, rect)
    dist = point_to_line_distance(X1, X2, point)

    return dist

def rectangle_to_closest_line_distance(B1, B2, G1, G2, rect):
    distB = rectangle_to_line_distance(B1, B2, rect)
    distG = rectangle_to_line_distance(G1, G2, rect)

    dist = distB
    blue_is_closer = True

    if distG < distB:
        dist = distG
        blue_is_closer = False

    return blue_is_closer, dist

def point_above_line(X1, X2, point):
    (x1, y1) = X1
    (x2, y2) = X2
    (x0, y0) = point

    k = (y1 - y2) / (x1 - x2)
    n = (y1 * (x1 - x2) - x1 * (y1 - y2)) / (x1 - x2)

    rightSide = k * x0 + n

    if y0 < rightSide:
        return True
    else:
        return False


def above_line(X1, X2, rect):
    # X1 and X2 are ends of the line
    (x1, y1) = X1
    (x2, y2) = X2

    closestPoint = closest_point_to_line(X1, X2, rect)
    (x0, y0) = closestPoint

    k = (y1 - y2) / (x1 - x2)
    n = (y1 * (x1 - x2) - x1 * (y1 - y2)) / (x1 - x2)

    rightSide = k*x0 + n

    if y0 < rightSide:
        A, B, C, D = rect_to_points(rect)
        aA = point_above_line(X1, X2, A)
        aB = point_above_line(X1, X2, B)
        aC = point_above_line(X1, X2, C)
        aD = point_above_line(X1, X2, D)
        if aA and aB and aC and aD:
            return True
        else:
            return False
    # if y0 < rightSide:
    #     return True
    # else:
    #     return False


def first_frame_rectangles(frame):
    frame_for_numbers = frame.copy()

    gray = cv2.cvtColor(frame_for_numbers, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    th_cont_img = th.copy()

    img, contours, hierarchy = cv2.findContours(th_cont_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    number_rect = []

    for contour in contours:
        center, size, angle = cv2.minAreaRect(contour)
        (x, y, w, h) = cv2.boundingRect(contour)
        width, height = size
        if width > 1 and width < 30 and height > 1 and height < 30:
            if width > 9 or height > 9:
                number_rect.append((x, y, w, h))

    return number_rect

def rect_to_points(rect):
    (x, y, w, h) = rect
    (x, y, w, h) = (int(x), int(y), int(w), int(h))
    A = (x, y)
    B = (x + w, y)
    C = (x, y + h)
    D = (x + w, y + h)

    return A, B, C, D

def get_int_tuple(tup):
    return (int(tup[0]), int(tup[1]))

def get_offset_values(A, B, C, D):
    ys = []
    xs = []
    ys.append(A[1])
    ys.append(B[1])
    ys.append(C[1])
    ys.append(D[1])

    xs.append(A[0])
    xs.append(B[0])
    xs.append(C[0])
    xs.append(D[0])

    ymin = min(ys)
    ymax = max(ys)
    xmin = min(xs)
    xmax = max(xs)

    return xmin, xmax, ymin, ymax

def pre_detect(frame, kernel, xmin, xmax, ymin, ymax):
    region = frame.copy()
    roi = region[ymin: ymax, xmin: xmax]

    blank_image = np.zeros((28, 28, 3), np.uint8)
    x_offset = (28 - roi.shape[0]) / 2
    y_offset = (28 - roi.shape[1]) / 2

    x_offset = int(round(x_offset))
    y_offset = int(round(y_offset))

    shape = (roi.shape[0], roi.shape[1])

    if roi.shape[0] > 25 or roi.shape[1] > 25:
        gray = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
        ret, th = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
        th = cv2.resize(th, (28, 28), interpolation=cv2.INTER_AREA)
        input_image = (np.expand_dims(th, 0))
        return shape, input_image

    blank_image[x_offset: x_offset + roi.shape[0], y_offset: y_offset + roi.shape[1]] = roi

    # ppp = blank_image.copy()
    # plt.imshow(ppp)
    # plt.show()

    gray = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    #th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    th = cv2.dilate(th, kernel, iterations = 1)
    th = cv2.erode(th, kernel, iterations = 1)
    #th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    # ppp = th.copy()
    # plt.imshow(ppp)
    # plt.show()

    th = cv2.resize(th, (28, 28), interpolation=cv2.INTER_AREA)
    input_image = (np.expand_dims(th, 0))

    return shape, input_image

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

    BlueA = lines_points[0][0]
    BlueB = lines_points[0][1]
    GreenA = lines_points[1][0]
    GreenB = lines_points[1][1]

    for contour in contours:
        center, size, angle = cv2.minAreaRect(contour)
        (x, y, w, h) = cv2.boundingRect(contour)
        width, height = size
        if width > 1 and width < 30 and height > 1 and height < 30:
            if width > 9 or height > 9:
                aboveBlue = above_line(BlueA, BlueB, (x, y, w, h))
                aboveGreen = above_line(GreenA, GreenB, (x, y, w, h))
                if aboveGreen or aboveBlue:
                    number_contours.append(contour) #not used
                    number_rect.append((x, y, w, h))

    img = frame.copy()
    #cv2.drawContours(img, boxes, -1, (0, 255, 255), 1)



    minDistBlue = 0
    minDistGreen = 0

    #print(len(number_rect))
    for rect in number_rect:
        #(A, B, C, D) = rect
        (x, y, w, h) = rect
        A = (x, y)
        B = (x + w, y)
        C = (x, y + h)
        D = (x + w, y + h)

        limit = 2

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

            ys = []
            xs = []
            ys.append(A[1])
            ys.append(B[1])
            ys.append(C[1])
            ys.append(D[1])

            xs.append(A[0])
            xs.append(B[0])
            xs.append(C[0])
            xs.append(D[0])

            ymin = min(ys)
            ymax = max(ys)
            xmin = min(xs)
            xmax = max(xs)

            region = frame.copy()
            kernel = np.ones((3, 3), np.uint8)

            if minDistBlue < minDistGreen:
                # Paint rectangle in blue
                if above_line(BlueA, BlueB, rect):
                    #cv2.drawContours(img, [box], 0, (255, 0, 0), 2)
                    cv2.rectangle(img, A, D, (255, 0, 0), 2)
                    # Add number
                    roi = region[ymin: ymax, xmin: xmax]

                    blank_image = np.zeros((28,28,3), np.uint8)
                    x_offset = (28-roi.shape[0])/2
                    y_offset = (28-roi.shape[1])/2

                    x_offset = int(round(x_offset))
                    y_offset = int(round(y_offset))

                    blank_image[x_offset: x_offset+roi.shape[0], y_offset: y_offset+roi.shape[1]] = roi

                    # ppp = blank_image.copy()
                    # plt.imshow(ppp)
                    # plt.show()

                    gray = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
                    ret, th = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
                    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
                    th = cv2.resize(th, (28, 28), interpolation=cv2.INTER_AREA)
                    input = (np.expand_dims(th, 0))

                    number = nn.predict(input)
                    #print(number)

                    #print("Number:")
                    #print(number)
                    suma += number
            else:
                if above_line(GreenA, GreenB, rect):
                    #cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
                    cv2.rectangle(img, A, D, (0, 255, 0), 2)
                    # Substract number
                    roi = region[ymin: ymax, xmin: xmax]

                    blank_image = np.zeros((28, 28, 3), np.uint8)
                    x_offset = (28 - roi.shape[0]) / 2
                    y_offset = (28 - roi.shape[1]) / 2

                    x_offset = int(round(x_offset))
                    y_offset = int(round(y_offset))

                    blank_image[x_offset: x_offset + roi.shape[0], y_offset: y_offset + roi.shape[1]] = roi

                    # ppp = blank_image.copy()
                    # plt.imshow(ppp)
                    # plt.show()

                    gray = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
                    ret, th = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
                    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
                    th = cv2.resize(th, (28, 28), interpolation=cv2.INTER_AREA)
                    input = (np.expand_dims(th, 0))

                    number = nn.predict(input)
                    #print(number)
                    suma -=  number

        else:
            # Rectangle is not close any line:
            #cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
            cv2.rectangle(img, A, D, (0, 0, 255), 2)


    return img, suma


def find_numbers_clean(frame, lines_points, suma, blue_numbers, green_numbers):
    suma = suma

    frame_for_numbers = frame.copy()

    gray = cv2.cvtColor(frame_for_numbers, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    th_cont_img = th.copy()

    img, contours, hierarchy = cv2.findContours(th_cont_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]

    number_contours = []
    number_rect = []

    BlueA = lines_points[0][0]
    BlueB = lines_points[0][1]
    GreenA = lines_points[1][0]
    GreenB = lines_points[1][1]

    for contour in contours:
        center, size, angle = cv2.minAreaRect(contour)
        (x, y, w, h) = cv2.boundingRect(contour)
        width, height = size
        if width > 1 and width < 30 and height > 1 and height < 30:
            if width > 9 or height > 9:
                aboveBlue = above_line(BlueA, BlueB, (x, y, w, h))
                aboveGreen = above_line(GreenA, GreenB, (x, y, w, h))
                if aboveGreen or aboveBlue:
                    number_contours.append(contour) #not used
                    number_rect.append((x, y, w, h))

    img = frame.copy()
    #cv2.drawContours(img, boxes, -1, (0, 255, 255), 1)



    minDistBlue = 0
    minDistGreen = 0

    #print(len(number_rect))
    for rect in number_rect:
        A, B, C, D = rect_to_points(rect)

        limit = 2

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

            xmin, xmax, ymin, ymax = get_offset_values(A, B, C, D)

            region = frame.copy()
            kernel = np.ones((3, 3), np.uint8)

            if minDistBlue < minDistGreen:
                # Paint rectangle in blue
                if above_line(BlueA, BlueB, rect):
                    # #cv2.drawContours(img, [box], 0, (255, 0, 0), 2)
                    # cv2.rectangle(img, A, D, (255, 0, 0), 2)
                    # # Add number
                    # roi = region[ymin: ymax, xmin: xmax]
                    #
                    # blank_image = np.zeros((28,28,3), np.uint8)
                    # x_offset = (28-roi.shape[0])/2
                    # y_offset = (28-roi.shape[1])/2
                    #
                    # x_offset = int(round(x_offset))
                    # y_offset = int(round(y_offset))
                    #
                    # blank_image[x_offset: x_offset+roi.shape[0], y_offset: y_offset+roi.shape[1]] = roi
                    #
                    # # ppp = blank_image.copy()
                    # # plt.imshow(ppp)
                    # # plt.show()
                    #
                    # gray = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
                    # ret, th = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
                    # th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
                    # th = cv2.resize(th, (28, 28), interpolation=cv2.INTER_AREA)
                    # input = (np.expand_dims(th, 0))
                    output_shape, input = pre_detect(frame, kernel, xmin, xmax, ymin, ymax)

                    number = nn.predict(input)
                    number = number[0]
                    addNumber = False
                    # Check blue_numbers
                    for input_image, last_blue_predicted_number, last_blue_roi_shape in blue_numbers:
                        if number == last_blue_predicted_number:
                            if output_shape[0] == last_blue_roi_shape[0] and output_shape[1] == last_blue_roi_shape[1]:
                                print("Isti su")
                                addNumber = False
                                break
                            else:
                                # Dodaj komparaciju slika jer se shape moze menjati za isti broj
                                print(input.shape)
                                print(input_image.shape)
                                #if np.logical_and(input_image, input):
                                if (input_image - input).any():
                                    addNumber = False
                                # Dodaj u listu
                                else:
                                    addNumber = True
                        else:
                            addNumber = True
                    if addNumber and number > 0:
                        if output_shape == (1, 28, 28, 3):
                            print("Strange shape")
                        else:
                            suma = suma + number
                            value = [input, number, output_shape]
                            blue_numbers.append(value)
                    # if number == last_blue_predicted_number:
                    #     if output_shape[0] == last_blue_roi_shape[0] and output_shape[1] == last_blue_roi_shape[1]:
                    #         print()
                    #     else:
                    #         suma = suma + number
                    #         last_blue_roi_shape = output_shape
                    #         last_blue_predicted_number = number
                    # else:
                    #     suma = suma + number
                    #     last_blue_roi_shape = output_shape
                    #     last_blue_predicted_number = number
                    print(number)
            else:
                if above_line(GreenA, GreenB, rect):
                    # #cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
                    # cv2.rectangle(img, A, D, (0, 255, 0), 2)
                    # # Substract number
                    # roi = region[ymin: ymax, xmin: xmax]
                    #
                    # blank_image = np.zeros((28, 28, 3), np.uint8)
                    # x_offset = (28 - roi.shape[0]) / 2
                    # y_offset = (28 - roi.shape[1]) / 2
                    #
                    # x_offset = int(round(x_offset))
                    # y_offset = int(round(y_offset))
                    #
                    # blank_image[x_offset: x_offset + roi.shape[0], y_offset: y_offset + roi.shape[1]] = roi
                    #
                    # # ppp = blank_image.copy()
                    # # plt.imshow(ppp)
                    # # plt.show()
                    #
                    # gray = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
                    # ret, th = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
                    # th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
                    # th = cv2.resize(th, (28, 28), interpolation=cv2.INTER_AREA)
                    # input = (np.expand_dims(th, 0))
                    output_shape, input = pre_detect(frame, kernel, xmin, xmax, ymin, ymax)
                    number = nn.predict(input)
                    number = number[0]
                    minusNumber = False
                    # Check green_numbers
                    for input_image, last_green_predicted_number, last_green_roi_shape in green_numbers:
                        if number == last_green_predicted_number:
                            if output_shape[0] == last_green_roi_shape[0] and output_shape[1] == last_green_roi_shape[1]:
                                print("Isti su")
                                minusNumber = False
                                break
                            else:
                                # Dodaj komparaciju slika jer se shape moze menjati
                                print(input.shape)
                                print(input_image.shape)
                                # plt.imshow(input_image)
                                # plt.show()
                                #if np.logical_and(input_image, input):
                                if (input_image - input).any():
                                    minusNumber = False
                                else:
                                    minusNumber = True
                        else:
                            minusNumber = True
                    if minusNumber and number > 0:
                        if output_shape == (1, 28, 28, 3):
                            print("Strange shape")
                        else:
                            suma = suma - number
                            value = [input, number, output_shape]
                            green_numbers.append(value)

                    # if number == last_green_predicted_number:
                    #     if output_shape[0] == last_green_roi_shape[0] and output_shape[1] == last_green_roi_shape[1]:
                    #         print()
                    #     else:
                    #         suma = suma - number
                    #         last_green_roi_shape = output_shape
                    #         last_green_predicted_number = number
                    # else:
                    #     suma = suma - number
                    #     last_green_roi_shape = output_shape
                    #     last_green_predicted_number = number
                    print(number)

        else:
            # Rectangle is not close any line:
            #cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
            cv2.rectangle(img, A, D, (0, 0, 255), 2)

    retVal = []
    retVal.append(blue_numbers)
    retVal.append(green_numbers)

    return img, suma, retVal