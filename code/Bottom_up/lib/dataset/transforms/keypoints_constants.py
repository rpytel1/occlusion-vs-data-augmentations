keypoint_names = {"coco": [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"],
    "mpii": ["right_ankle", "right_knee", "right_hip", "left_hip", "left_knee", "left_ankle", "pelvis", "thorax",
             "upper_neck", "head_top",
             "right_wrist", 'right_elbow', "right_shoulder", "left_shoulder", "left_elbow", "left_wrist"]}

part_mapping = {"coco": {"head": [0, 1, 2, 3, 4], "left_arm": [5, 7, 9], "right_arm": [6, 8, 10], "hip": [11, 12],
                         "left_leg": [11, 13, 15], "right_leg": [12, 14, 16], "left_side": [1, 3, 5, 7, 9, 11, 13, 15],
                         "right_side": [0, 2, 4, 6, 8, 10, 12, 14, 16],
                         "upper_body": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                         "lower_body": [11, 12, 13, 14, 15, 16],
                         "corpus": [5, 6, 11, 12]},
                "mpii": {"head": [8, 9], "left_arm": [13, 14, 15], "right_arm": [10, 11, 12], "hip": [2, 3],
                         "left_leg": [3, 4, 5], "right_leg": [0, 1, 2], "left_side": [3, 4, 5, 13, 14, 15],
                         "right_side": [0, 1, 2, 10, 11, 12], "upper_body": [7, 8, 9, 10, 11, 12, 13, 14, 15],
                         "lower_body": [0, 1, 2, 3, 4, 5, 6], "corpus": [2, 3, 12, 13]}}

advanced_parts = {"coco": {
    "head": ["nose", "right_eye", "right_ear", "left_eye", "left_ear"],
    "left_side": ["left_hip", "left_knee", "left_ankle", "left_elbow", "left_wrist", "left_shoulder", "left_ear",
                  "left_eye"],
    "right_side": ["right_hip", "right_knee", "right_ankle", "right_elbow", "right_wrist", "right_shoulder",
                   "right_ear",
                   "right_eye"],
    "upper_body": ["left_ear", "right_ear", "left_eye", "right_eye", "nose", "left_shoulder", "right_shoulder",
                   "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip"],
    "lower_body": ["left_ankle", "left_knee", "left_hip", "right_hip", "right_knee", "right_ankle"],
    "corpus": ["left_hip", "right_hip", "right_shoulder", "left_shoulder"]},
    "mpii": {
        "head": ["upper_neck", "head_top"],
        "left_side": ["left_hip", "left_knee", "left_ankle", "left_elbow", "left_wrist", "left_shoulder"],
        "right_side": ["right_hip", "right_knee", "right_ankle", "right_elbow", "right_wrist", "right_shoulder"],
        "upper_body": ["head_top", "upper_neck", "left_shoulder", "right_shoulder",
                       "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip"],
        "lower_body": ["left_ankle", "left_knee", "left_hip", "right_hip", "right_knee", "right_ankle"],
        "corpus": ["left_hip", "right_hip", "right_shoulder", "left_shoulder"]}}
