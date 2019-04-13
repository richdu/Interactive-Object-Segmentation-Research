import numpy as np
import cv2

BG_COLOR = 1
FG_COLOR = 2

def get_positive_region(actual, predicted):
    return actual&(~predicted)

def get_positive_input(actual, predicted):
    region = get_positive_region(actual, predicted)
    idx = np.nonzero(region)
    rand_idx = np.random.randint(len(idx[0]))
    x, y = idx[0][rand_idx], idx[1][rand_idx]
    new_mask = np.zeros(actual.shape) # new mask
    cv2.circle(new_mask, (y, x), 10, 1, -1)
    positive_input = new_mask.astype(int)&actual
    return FG_COLOR*positive_input

def get_positive_line(actual, predicted):
    region = get_positive_region(actual, predicted)
    idx = np.nonzero(region)
    rand_idx_1 = np.random.randint(len(idx[0]))
    rand_idx_2 = np.random.randint(len(idx[0]))
    x1, y1 = idx[0][rand_idx_1], idx[1][rand_idx_1]
    x2, y2 = idx[0][rand_idx_2], idx[1][rand_idx_2]
    new_mask = np.zeros(actual.shape) # new mask
    cv2.line(new_mask, (y1, x1), (y2, x2), 1, 10)
    positive_line = new_mask.astype(int)&actual
    return FG_COLOR*positive_line

def get_negative_region(actual, predicted):
    return (~actual)&predicted

def get_negative_input(actual, predicted):
    region = get_negative_region(actual, predicted)
    idx = np.nonzero(region)
    rand_idx = np.random.randint(len(idx[0]))
    x, y = idx[0][rand_idx], idx[1][rand_idx]
    new_mask = np.zeros(actual.shape) # new mask
    cv2.circle(new_mask, (y, x), 10, 1, -1)
    negative_input = new_mask.astype(int)&(~actual)
    return BG_COLOR*negative_input

def get_negative_line(actual, predicted):
    region = get_negative_region(actual, predicted)
    idx = np.nonzero(region)
    rand_idx_1 = np.random.randint(len(idx[0]))
    rand_idx_2 = np.random.randint(len(idx[0]))
    x1, y1 = idx[0][rand_idx_1], idx[1][rand_idx_1]
    x2, y2 = idx[0][rand_idx_2], idx[1][rand_idx_2]
    new_mask = np.zeros(actual.shape) # new mask
    cv2.line(new_mask, (y1, x1), (y2, x2), BG_COLOR, 10)
    negative_line = new_mask.astype(int)&(~actual)
    return BG_COLOR*negative_line
