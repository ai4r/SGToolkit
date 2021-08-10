import numpy as np
from sklearn.preprocessing import normalize

from utils.data_utils import convert_dir_vec_to_pose, dir_vec_pairs


def convert_pose_to_line_segments(pose):
    line_segments = np.zeros((len(dir_vec_pairs) * 2, 3))
    for j, pair in enumerate(dir_vec_pairs):
        line_segments[2 * j] = pose[pair[0]]
        line_segments[2 * j + 1] = pose[pair[1]]

    line_segments[:, [1, 2]] = line_segments[:, [2, 1]]  # swap y, z
    line_segments[:, 2] = -line_segments[:, 2]
    return line_segments


def convert_dir_vec_to_line_segments(dir_vec):
    joint_pos = convert_dir_vec_to_pose(dir_vec)
    line_segments = np.zeros((len(dir_vec_pairs) * 2, 3))
    for j, pair in enumerate(dir_vec_pairs):
        line_segments[2 * j] = joint_pos[pair[0]]
        line_segments[2 * j + 1] = joint_pos[pair[1]]

    line_segments[:, [1, 2]] = line_segments[:, [2, 1]]  # swap y, z
    line_segments[:, 2] = -line_segments[:, 2]
    return line_segments

