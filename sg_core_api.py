"""
    Wrapper for sg_core modules.
    If sg_core changes, only this file should be changed.
"""
import pathlib

this_dir_path = pathlib.Path(__file__).parent

# define model file path
model_path = this_dir_path.joinpath('sg_core', 'output', 'sgtoolkit', 'multimodal_context_checkpoint_best.bin')
assert model_path.exists(), "model file ({}) does not exists:".format(str(model_path))
model_file_name = str(model_path)

# add sg_core in path
import sys
import os

sg_core_scripts_path = this_dir_path.joinpath('sg_core', 'scripts')
sg_core_path = this_dir_path.joinpath('sg_core')
gentle_path = this_dir_path.joinpath('gentle')
gentle_ext_path = this_dir_path.joinpath('gentle', 'ext')
google_key_path = this_dir_path.joinpath('sg_core', 'google-key.json')
sys.path.append(str(sg_core_scripts_path))
sys.path.append(str(sg_core_path))
sys.path.append(str(gentle_path))
sys.path.append(str(gentle_ext_path))
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(google_key_path)

from sg_core.scripts.gesture_generator import GestureGenerator
import numpy as np


def get_gesture_generator():
    audio_cache_path = './cached_wav'
    return GestureGenerator(model_file_name, audio_cache_path)


def convert_pose_coordinate_for_ui(pose_mat):
    return flip_y_axis_of_pose(pose_mat)


def convert_pose_coordinate_for_model(constraint_mat):
    mask_col = constraint_mat[:, -1]
    mask_col = mask_col[:, np.newaxis]
    pose_mat = constraint_mat[:, :-1]
    pose_mat = flip_y_axis_of_pose(pose_mat)
    return np.hstack((pose_mat, mask_col))


def convert_pose_coordinate_for_ui_for_motion_library(motions_cursor):
    converted = []
    for motion in list(motions_cursor):
        motion_mat = np.array(motion['motion'])
        motion['motion'] = convert_pose_coordinate_for_ui(motion_mat).tolist()
        converted.append(motion)

    return converted


def convert_pose_coordinate_for_ui_for_rule_library(rules_cursor):
    converted = []
    for rule in list(rules_cursor):
        if (rule['motion_info'] == []):
            continue
        motion_mat = np.array(rule['motion_info'][0]['motion'])
        rule['motion_info'][0]['motion'] = convert_pose_coordinate_for_ui(motion_mat).tolist()
        converted.append(rule)

    return converted


def flip_y_axis_of_pose(pose_mat):
    n_frame = pose_mat.shape[0]
    pose_mat = np.reshape(pose_mat, (n_frame, -1, 3))
    pose_mat[:, :, 1] = -pose_mat[:, :, 1]
    pose_mat = np.reshape(pose_mat, (n_frame, -1))
    return pose_mat
