from .keypoints_constants import keypoint_names


def extract_keypoints(keypoint_arr, dataset):
    new_elem = {}
    for i, joint in enumerate(keypoint_arr):
        if joint[0] != 0 and joint[1] != 0:
            new_elem[keypoint_names[dataset][i]] = (joint[0], joint[1])
    return [new_elem]
