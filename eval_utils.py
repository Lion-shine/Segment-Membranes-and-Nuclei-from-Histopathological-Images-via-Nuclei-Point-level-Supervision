import numpy as np
import cv2

from collections import Counter
def sort_edge(image):
    # if image!=None:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours[1]
    else:
        gray = image
        ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours[0]


# else:
# return 0

def find_single_contour_centre_point(single_contour):
    max_x = max(single_contour[:, :, 0])
    min_x = min(single_contour[:, :, 0])
    max_y = max(single_contour[:, :, 1])
    min_y = min(single_contour[:, :, 1])
    point_x = int((max_x + min_x) / 2)
    point_y = int((max_y + min_y) / 2)
    point = [point_x, point_y]
    return point


def calculate_distence(pred_point, label_point):
    # pred_point_x=pred_point[0]
    # pred_point_y = pred_point[1]
    # label_point_x=label_point[0]
    # label_point_y = label_point[1]
    return np.sqrt(np.sum((np.array(pred_point) - np.array(label_point)) ** 2))


def find_true_postive_point(pred_contours_centre_point_list, label_contours_centre_point_list, threshold_value):
    # positive_point_number = 0
    # false_positive_number = 0

    distence_array = np.ones(len(pred_contours_centre_point_list)) * 1000000
    label_distence_array = np.ones(len(label_contours_centre_point_list)) * 1000000
    for i in range(len(pred_contours_centre_point_list)):
        pred_point = pred_contours_centre_point_list[i]
        for j in range(len(label_contours_centre_point_list)):
            label_point = label_contours_centre_point_list[j]
            distence = calculate_distence(pred_point, label_point)
            if distence <= distence_array[i]:
                distence_array[i] = distence
    for i in range(len(label_contours_centre_point_list)):
        label_point = label_contours_centre_point_list[i]
        for j in range(len(pred_contours_centre_point_list)):
            pred_point = pred_contours_centre_point_list[j]
            distence = calculate_distence(pred_point, label_point)
            if distence <= label_distence_array[i]:
                label_distence_array[i] = distence
    positive_point_number = len(np.where(distence_array <= threshold_value)[0])
    false_positive_number = len(np.where(distence_array > threshold_value)[0])
    false_negitive_number = len(np.where(label_distence_array > threshold_value)[0])
    return positive_point_number, false_positive_number, false_negitive_number


def find_true_postive_point_json(test_center_coords, test_labels, label_center_coords, labels_labels, threshold_value):
    # positive_point_number = 0
    # false_positive_number = 0
    distence_array = np.ones(len(test_labels)) * 1000000
    label_distence_array = np.ones(len(labels_labels)) * 1000000
    for i in range(len(test_labels)):
        pred_point = test_center_coords[i]
        for j in range(len(labels_labels)):
            label_point = label_center_coords[j]
            if labels_labels[j] == test_labels[i]:
                distence = calculate_distence(pred_point, label_point)
                if distence <= distence_array[i]:
                    distence_array[i] = distence
    for i in range(len(labels_labels)):
        label_point = label_center_coords[i]
        for j in range(len(test_labels)):
            pred_point = test_center_coords[j]
            if test_labels[j] == labels_labels[i]:
               distence = calculate_distence(pred_point, label_point)
               if distence <= label_distence_array[i]:
                  label_distence_array[i] = distence
    positive_point_number = len(np.where(distence_array <= threshold_value)[0])
    false_positive_number = len(np.where(distence_array > threshold_value)[0])
    false_negitive_number = len(np.where(label_distence_array > threshold_value)[0])
    return positive_point_number, false_positive_number, false_negitive_number

def find_true_postive_point(test_center_coords, test_labels, label_center_coords, labels_labels, threshold_value):
    # positive_point_number = 0
    # false_positive_number = 0
    distence_array = np.ones(test_labels.shape[0]) * 1000000
    label_distence_array = np.ones(labels_labels.shape[0]) * 1000000
    for i in range(test_labels.shape[0]):
        pred_point = test_center_coords[i]
        for j in range(labels_labels.shape[0]):
            label_point = label_center_coords[j]
            if labels_labels[j] == test_labels[i]:
                distence = calculate_distence(pred_point, label_point)
                if distence <= distence_array[i]:
                    distence_array[i] = distence
    for i in range(labels_labels.shape[0]):
        label_point = label_center_coords[i]
        for j in range(test_labels.shape[0]):
            pred_point = test_center_coords[j]
            if test_labels[j] == labels_labels[i]:
               distence = calculate_distence(pred_point, label_point)
               if distence <= label_distence_array[i]:
                  label_distence_array[i] = distence
    positive_point_number = len(np.where(distence_array <= threshold_value)[0])
    false_positive_number = len(np.where(distence_array > threshold_value)[0])
    false_negitive_number = len(np.where(label_distence_array > threshold_value)[0])
    return positive_point_number, false_positive_number, false_negitive_number

def find_true_positive_and_count(pred_mask, label_mask, threshold_value):
    pred_contours_list = sort_edge(pred_mask)
    label_contours_list = sort_edge(label_mask)
    pred_contours_centre_point_list = []
    label_contours_centre_point_list = []
    for i in range(len(pred_contours_list)):
        single_contour = pred_contours_list[i]
        point = find_single_contour_centre_point(single_contour)
        pred_contours_centre_point_list.append(point)
    for i in range(len(label_contours_list)):
        single_contour = label_contours_list[i]
        point = find_single_contour_centre_point(single_contour)
        label_contours_centre_point_list.append(point)
    positive_point_number, false_positive_number, false_negitive_number = find_true_postive_point(
        pred_contours_centre_point_list,
        label_contours_centre_point_list,
        threshold_value)
    return positive_point_number, false_positive_number, false_negitive_number
