import numpy as np
import imageio
import math
import cv2
from ast import literal_eval as make_tuple
from matplotlib import pyplot as plt
import json


def hough_line(img, angle_step=1, lines_are_white=True, value_threshold=100):
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_step))

    width, height = img.shape
    diag_len = int(round(math.sqrt(width * width + height * height)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)
    # print(num_thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint8)
    # (row, col) indexes to edges
    are_edges = img > value_threshold if lines_are_white else img < value_threshold
    y_idxs, x_idxs = np.nonzero(are_edges)


    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
            # print(rho)
            accumulator[rho, t_idx] += 1

    return accumulator, thetas, rhos

def hough_circle(ori_img, img, value_threshold=200):
    # Rho and Theta ranges
    thetas = np.arange(0,180)* math.pi/ 180
    num_thetas = len(thetas)
    width, height = img.shape
    radius_min = 10
    radius_max = int(np.nanmax([width, height]) / 2)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    accumulator = np.zeros((width + radius_max, height + radius_max, radius_max))
    are_edges = img > value_threshold if True else img < value_threshold
    y_idxs, x_idxs = np.nonzero(are_edges)
    for i in range(len(x_idxs)):
        # print(i, len(x_idxs))
        x = x_idxs[i]
        y = y_idxs[i]
        pos = np.nanmax([x, y])
        for radius in range(20, radius_max-pos):
            a = int(x - radius * cos_t[0])
            b = int(y - radius * sin_t[0])
            a2 = int(x + radius * cos_t[0])
            b2 = int(y + radius * sin_t[0])
            
            accumulator[a, b, radius] +=1
            # print(x, radius * cos_t[0])
            # print(a2,b2, radius)
            accumulator[a2, b2, radius] +=1
            a = int(x - radius * cos_t[179])
            b = int(y - radius * sin_t[179])
            a2 = int(x + radius * cos_t[179])
            b2 = int(y + radius * sin_t[179])
            accumulator[a, b, radius] +=1
            accumulator[a2, b2, radius] +=1
    return accumulator

def draw_detect_dart(image, accumulator):
    circleCoordinates = np.argwhere(accumulator>3.7)
    result = []
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(gray_image)
    dart_cascade = cv2.CascadeClassifier('dartcascade/cascade.xml')
    darts = dart_cascade.detectMultiScale(gray_image, 1.1, 1,
                                        0 | cv2.CASCADE_SCALE_IMAGE,
                                        (50, 50), (500, 500))
    for (x, y, w, h) in darts:
        top_left = (x, y)
        bottom_right = (x+w, y+h)
        result.append({"top_left": top_left, "bottom_right": bottom_right})
        for x1,y1,r in circleCoordinates:       
            if(x1 in range(x, x+w) and y1 in range(y, y+h)):
                cv2.rectangle(image,top_left,bottom_right,(0,255,0),2)
    return image, result

def draw_truth_darts(image, json_file_name):
    result = []
    fp = open(json_file_name)
    darts = json.loads(fp.read())
    for dart in darts:
        top_left = make_tuple(dart['top_left'])
        bottom_right = make_tuple(dart['bottom_right'])
        result.append({"top_left": top_left, "bottom_right": bottom_right})
        cv2.rectangle(image,top_left,bottom_right,(0,0,255),2)
    return image, result

def calculate_truth_false(bounds_truth, bounds_detected):
    tp = 0
    fp = 0
    fn = 0
    for turth in bounds_truth:
        temp = 0
        for detected in bounds_detected:
            a = (turth['top_left'][0] - detected['top_left'][0])** 2
            b = (turth['top_left'][1] - detected['top_left'][1])** 2
            c = (turth['bottom_right'][0] - detected['bottom_right'][0])** 2
            d = (turth['bottom_right'][1] - detected['bottom_right'][1])** 2
            if a+b+c+d<1000:
                tp += 1
            else:
                temp += 1
        if temp == len(bounds_detected):
            fn += 1

    for detected in bounds_detected:
        temp = 0
        for turth in bounds_truth:
            a = (turth['top_left'][0] - detected['top_left'][0])** 2
            b = (turth['top_left'][1] - detected['top_left'][1])** 2
            c = (turth['bottom_right'][0] - detected['bottom_right'][0])** 2
            d = (turth['bottom_right'][1] - detected['bottom_right'][1])** 2
            if a+b+c+d>1000:
                temp += 1
        if temp >= len(bounds_truth):
            fp += 1
    return tp, fp, fn

def calculate_f1_score(tp, fp, fn):
    f1 = (float(2)*tp)/(float(2)*tp+fn+fp)
    return f1



if __name__ == '__main__':
    image_folder = "../images/dart/"
    turth_json_folder = "truth_bounded/"
    result_folder = "result/"
    image_list = ['dart0', 'dart1', 'dart2', 'dart3', 'dart4', 'dart5', 'dart6', 'dart7', 'dart8', 'dart9', 'dart10', 'dart11', 'dart12', 'dart13', 'dart14', 'dart15', 'dart16']
    f1_total = 0
    average = 0
    for image_name in image_list:
        
        
        image = cv2.imread(image_folder + image_name + ".jpg")
        image_truth, bounds_truth = draw_truth_darts(image, turth_json_folder+image_name+".json")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_gaussian = cv2.GaussianBlur(gray_image,(3,3),0)

        #sobel
        img_sobelx = cv2.Sobel(img_gaussian,cv2.CV_8U,1,0,ksize=3)
        img_sobely = cv2.Sobel(img_gaussian,cv2.CV_8U,0,1,ksize=3)
        img_sobel = img_sobelx + img_sobely

        #hough transform
        accumulator = hough_circle(image, img_sobel)

        #adaboost
        image_decteted, bounds_detected = draw_detect_dart(image_truth, accumulator)
        cv2.imwrite(result_folder + image_name + "_detected.jpg", image_decteted)
        tp, fp, fn = calculate_truth_false(bounds_truth, bounds_detected)
        f1 = calculate_f1_score(tp, fp, fn)
        f1_total += f1
    average = f1_total / 16

    