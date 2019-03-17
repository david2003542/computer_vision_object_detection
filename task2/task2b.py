import json
from ast import literal_eval as make_tuple

import cv2
import matplotlib.pyplot as plt


def draw_detect_dart(image):
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
    image_list = ['dart0', 'dart1', 'dart2', 'dart3', 'dart4', 'dart5', 'dart6', 'dart7', 'dart8', 'dart9', 'dart10', 'dart11', 'dart12', 'dart13', 'dart14', 'dart15']
    f1_total = 0
    average = 0
    for image_name in image_list:
        image = cv2.imread(image_folder + image_name + ".jpg")
        image_truth, bounds_truth = draw_truth_darts(image, turth_json_folder+image_name+".json")
        image_decteted, bounds_detected = draw_detect_dart(image_truth)
        cv2.imwrite(result_folder + image_name + "_detected.jpg", image_decteted)
        tp, fp, fn = calculate_truth_false(bounds_truth, bounds_detected)
        f1 = calculate_f1_score(tp, fp, fn)
        print(f1)
        f1_total += f1
    average = float(f1_total) / 16
    print(average)
    
    
        






