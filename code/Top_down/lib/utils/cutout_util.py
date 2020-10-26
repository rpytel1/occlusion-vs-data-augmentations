import cv2
import numpy as np
import json
import math
import shutil
import os

from lib.utils.keypoints_constants import part_mapping, advanced_parts, keypoint_names


def get_image_mean_color(img):
    return img.mean(axis=0).mean(axis=0)


def get_xy(center, widths, img_dims):
    x, y = center
    width_x, width_y = widths
    h, w = img_dims
    y1 = int(np.clip(y - width_y / 2, 0, h))
    y2 = int(np.clip(y + width_y / 2, 0, h))
    x1 = int(np.clip(x - width_x / 2, 0, w))
    x2 = int(np.clip(x + width_x / 2, 0, w))
    return x1, x2, y1, y2


def rect_cutout(image, x_center, y_center, x_length, y_length, color):
    x1, x2, y1, y2 = get_xy((x_center, y_center), (x_length, y_length), image.shape[:-1])
    image[y1:y2, x1:x2] = color
    return image


def circle_cutout(image, x_center, y_center, radius, color):
    h, w = image.shape[:-1]
    if x_center >= 0 and x_center < image.shape[0] and y_center >= 0 and y_center < image.shape[1]:
        x_center = max(x_center, 0)
        y_center = max(y_center, 0)
        x, y = np.ogrid[-y_center:h - y_center, -x_center:w - x_center]
        mask = x * x + y * y <= radius * radius

        image[mask] = color
    return image


def circle_blur(image, x_center, y_center, radius):
    # 9 times std (as it is 3 times
    blurred_img = cv2.GaussianBlur(image, (2*radius-1, 2*radius-1), 0)
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask = circle_cutout(mask, x_center, y_center, radius, np.array([255, 255, 255]))
    out = np.where(mask == np.array([255, 255, 255]), blurred_img, image)

    return out


def blur(image, x_center, y_center, x_length, y_length):
    blurred_img = cv2.GaussianBlur(image, (31, 31), 0)
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask = rect_cutout(mask, x_center, y_center, x_length, y_length, [255, 255, 255])
    out = np.where(mask == np.array([255, 255, 255]), blurred_img, image)

    return out


def get_image_dict(annotations):
    images_list = [(elem['id'], elem['file_name']) for elem in annotations['images']]
    return dict(images_list)


def extract_json(annotations, dataset):
    keypoints_dict = {}
    if dataset == "coco":
        images_dict = get_image_dict(annotations)
        for elem in annotations['annotations']:
            file_name = images_dict[elem['image_id']]
            if file_name not in keypoints_dict:
                keypoints_dict[file_name] = []
            keypoints = extract_keypoints(elem['keypoints'], dataset)
            if keypoints != {}:
                keypoints_dict[file_name].append(keypoints)
    else:
        for elem in annotations:
            if elem['image'] not in keypoints_dict:
                keypoints_dict[elem['image']] = []

            keypoints = extract_keypoints(elem['joints'], "mpii")
            if keypoints != {}:
                keypoints_dict[elem['image']].append(keypoints)

    return keypoints_dict


def extract_keypoints(keypoints_json, dataset):
    keypoints_pos = {}
    if dataset == "coco":
        for keypoint_id in range(len(keypoints_json) // 3):
            if keypoints_json[keypoint_id * 3 + 2] != 0:
                keypoints_pos[keypoint_names[dataset][keypoint_id]] = (
                    keypoints_json[keypoint_id * 3], keypoints_json[keypoint_id * 3 + 1])
    else:
        keypoints_pos = {}
        for i, elem in enumerate(keypoints_json):
            keypoints_pos[keypoint_names[dataset][i]] = (elem[0], elem[1])

    return keypoints_pos


def get_centerpoint(keypoints, key1, key2):
    keypoint1 = np.array(keypoints[key1])
    keypoint2 = np.array(keypoints[key2])
    return np.mean(np.array([keypoint1, keypoint2]), axis=0)


def get_shifted_point(keypoints, key1, key2, ratio):
    keypoint1 = np.array(keypoints[key1])
    keypoint2 = np.array(keypoints[key2])
    dist = ratio * abs(keypoint1 - keypoint2)
    return keypoint2 - dist


def get_dist(keypoints, key1, key2, key3=None):
    keypoint1 = np.array(keypoints[key1])
    keypoint2 = np.array(keypoints[key2])
    dist = abs(keypoint1 - keypoint2)
    min_dist = np.flip(0.25 * dist)
    dist = np.max([dist, min_dist], axis=0)
    if key3 and key3 in keypoints.keys():
        keypoint3 = np.array(keypoints[key3])
        dist2 = abs(keypoint2 - keypoint3)
        dist = np.array([dist, dist2]).max(axis=0)
    return dist


def calculate_center_with_dist(keypoints, chosen_keypoints):
    chosen_keypoints_pos = []
    for key in chosen_keypoints:
        if key in keypoints:
            chosen_keypoints_pos.append(keypoints[key])
    if len(chosen_keypoints_pos) > 1:
        chosen_keypoints_arr = np.array(chosen_keypoints_pos)
        min_pos = np.min(chosen_keypoints_arr, axis=0)
        max_pos = np.max(chosen_keypoints_arr, axis=0)
        dists = (max_pos - min_pos)
        center_point = np.mean(np.array([max_pos, min_pos]), axis=0)

        return center_point, dists
    return (0, 0), (0, 0)


def get_cords_with_width(keypoints, part, dataset):
    """
    Method using various heuristics to return coordinates of middle point for box
    :param img: image on which
    :param keypoints: dictionary of positions of various keypoints
    :param part: part which will be occluded

    :return: coordinates of a center point of box, width in X and Y of a box
    """
    if part == "head":
        if dataset == "coco":
            out = calculate_center_with_dist(keypoints, advanced_parts[dataset]['head'])
            return out[0], (1.2 * out[1][0], 1.2 * out[1][1])
        else:
            if "head_top" in keypoints.keys() and "upper_neck" in keypoints.keys():
                dists = get_dist(keypoints, "head_top", "upper_neck")
                return get_centerpoint(keypoints, "head_top", "upper_neck"), (2*dists[0], dists[1])
    elif part == "hip":
        if "left_hip" in keypoints.keys() and "right_hip" in keypoints.keys():
            return get_centerpoint(keypoints, "left_hip", "right_hip"), 2 * get_dist(keypoints, "left_hip",
                                                                                     "right_hip")
    elif part == "left_arm":
        if "left_wrist" in keypoints.keys() and "left_elbow" in keypoints.keys() and "left_shoulder" in keypoints.keys():
            return calculate_center_with_dist(keypoints, ["left_wrist", "left_elbow", "left_shoulder"])
        elif "left_wrist" in keypoints.keys() and "left_elbow" in keypoints.keys():
            return keypoints['left_elbow'], 2 * get_dist(keypoints, "left_wrist", "left_elbow", "left_shoulder")

        elif "left_shoulder" in keypoints.keys() and "left_elbow" in keypoints.keys():
            return keypoints['left_elbow'], 2 * get_dist(keypoints, "left_shoulder", "left_elbow", "left_wrist")

        elif "left_shoulder" in keypoints.keys() and "left_wrist" in keypoints.keys():
            return get_centerpoint(keypoints, "left_shoulder", "left_wrist"), get_dist(keypoints, "left_wrist",
                                                                                       "left_shoulder")
    elif part == "right_arm":
        if "right_wrist" in keypoints.keys() and "right_elbow" in keypoints.keys() and "right_shoulder" in keypoints.keys():
            return calculate_center_with_dist(keypoints, ["right_wrist", "right_elbow", "right_shoulder"])
        elif "right_wrist" in keypoints.keys() and "right_elbow" in keypoints.keys():
            return keypoints['right_elbow'], 2 * get_dist(keypoints, "right_wrist", "right_elbow", "right_shoulder")

        elif "right_shoulder" in keypoints.keys() and "right_elbow" in keypoints.keys():
            return keypoints['right_elbow'], 2 * get_dist(keypoints, "right_shoulder", "right_elbow")

        elif "right_shoulder" in keypoints.keys() and "right_wrist" in keypoints.keys():
            return get_centerpoint(keypoints, "right_shoulder", "right_wrist"), get_dist(keypoints, "right_wrist",
                                                                                         "right_shoulder")
    elif part == "left_leg":
        if "left_hip" in keypoints.keys() and "left_knee" in keypoints.keys() and "left_ankle" in keypoints.keys():
            return calculate_center_with_dist(keypoints, ["left_ankle", "left_knee", "left_hip"])

        elif "left_hip" in keypoints.keys() and "left_knee" in keypoints.keys():
            return keypoints['left_knee'], 2 * get_dist(keypoints, "left_hip", "left_knee", "left_ankle")

        elif "left_ankle" in keypoints.keys() and "left_knee" in keypoints.keys():
            return keypoints['left_knee'], 2 * get_dist(keypoints, "left_ankle", "left_knee")

        elif "left_ankle" in keypoints.keys() and "left_hip" in keypoints.keys():
            return get_centerpoint(keypoints, "left_ankle", "left_hip"), get_dist(keypoints, "left_ankle", "left_hip")

    elif part == "right_leg":
        if "right_hip" in keypoints.keys() and "right_knee" in keypoints.keys() and "right_ankle" in keypoints.keys():
            return calculate_center_with_dist(keypoints, ["right_ankle", "right_knee", "right_hip"])

        elif "right_hip" in keypoints.keys() and "right_knee" in keypoints.keys():
            return keypoints['right_knee'], 2 * get_dist(keypoints, "right_hip", "right_knee", "right_ankle")

        elif "right_ankle" in keypoints.keys() and "right_knee" in keypoints.keys():
            return keypoints['right_knee'], 2 * get_dist(keypoints, "right_ankle", "right_knee", "right_ankle")

        elif "right_ankle" in keypoints.keys() and "right_hip" in keypoints.keys():
            return get_centerpoint(keypoints, "right_ankle", "right_hip"), get_dist(keypoints, "right_ankle",
                                                                                    "right_hip", "right_knee")
    elif part == "corpus":
        return calculate_center_with_dist(keypoints, advanced_parts[dataset]["corpus"])
    elif part == "upper_body":
        return calculate_center_with_dist(keypoints, advanced_parts[dataset]["upper_body"])
    elif part == "lower_body":
        return calculate_center_with_dist(keypoints, advanced_parts[dataset]["lower_body"])
    elif part == "right_side":
        return calculate_center_with_dist(keypoints, advanced_parts[dataset]["right_side"])
    elif part == "left_side":
        return calculate_center_with_dist(keypoints, advanced_parts[dataset]["left_side"])

    return None  # Fallback option


def blurr_img(img, keypoints_list, part, dataset, width=6):
    concetrated_cutout = part not in part_mapping[dataset].keys()

    for keypoints in keypoints_list:
        if concetrated_cutout:
            if part in keypoints:
                # width = 6  # As it is in the ground truth
                x, y = keypoints[part]

                img = circle_blur(img, x, y, width)

                return img, (x, y), (width, width)
        else:
            cords = get_cords_with_width(keypoints, part, dataset)
            if cords:
                (x, y), (width_x, width_y) = cords

                img = blur(img, x, y, width_x, width_y)

                return img, (x, y), (width_x, width_y)
    return img, (0, 0), (0, 0)


def cutout_img(img, keypoints_list, part, mean_coloring, dataset, width = 6):
    color = [0, 0, 0]
    concetrated_cutout = part not in part_mapping[dataset].keys()
    if mean_coloring:
        color = get_image_mean_color(img)

    for keypoints in keypoints_list:
        if concetrated_cutout:
            if part in keypoints:
                x, y = keypoints[part]
                img = circle_cutout(img, x, y, width, color)
                return img, (x, y), (width, width)
        else:
            cords = get_cords_with_width(keypoints, part, dataset)
            if cords:
                (x, y), (width_x, width_y) = cords

                img = rect_cutout(img, x, y, width_x, width_y, color)

                return img, (x, y), (width_x, width_y)
    return img, (0, 0), (0, 0)


def copy_image_files(path, dst):
    if not os.path.exists(dst):
        os.mkdir(dst)
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            shutil.copyfile(path + file, dst + file)


def remove_annotation(keypoint_arr, joints_vis, center_pos, widths):
    for id in range(keypoint_arr.shape[0]):
        pos = (keypoint_arr[id][0], keypoint_arr[id][1])
        if is_within_box(pos, center_pos, widths):
            keypoint_arr[id] = [0, 0, 0]
            joints_vis[id] = [0, 0, 0]

    return keypoint_arr, joints_vis


def is_within_box(pos, center_pos, widths):
    x_lower = center_pos[0] - widths[0]
    x_upper = center_pos[0] + widths[0]
    y_lower = center_pos[1] - widths[1]
    y_upper = center_pos[1] + widths[1]
    if widths[0] == 0 or widths[0] == 0:
        return False
    return x_lower < pos[0] <= x_upper and y_lower <= pos[1] <= y_upper
