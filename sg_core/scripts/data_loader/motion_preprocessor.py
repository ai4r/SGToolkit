import random

import numpy as np


class MotionPreprocessor:
    def __init__(self, skeletons, mean_pose):
        self.skeletons = np.array(skeletons)
        self.mean_pose = np.array(mean_pose).reshape(-1, 3)
        self.filtering_message = "PASS"

    def get(self):
        assert (self.skeletons is not None)

        # filtering
        if self.skeletons != []:
            verbose = False
            if self.check_frame_diff(verbose):
                self.skeletons = []
                self.filtering_message = "frame_diff"
            # elif self.check_spine_angle(verbose):
            #     self.skeletons = []
            #     self.filtering_message = "spine_angle"
            elif self.check_static_motion(verbose):
                if random.random() < 0.9:  # keep 10%
                    self.skeletons = []
                    self.filtering_message = "motion_var"

        if self.skeletons != []:
            self.skeletons = self.skeletons.tolist()
            for i, frame in enumerate(self.skeletons):
                # assertion: missing joints
                assert not np.isnan(self.skeletons[i]).any()

        return self.skeletons, self.filtering_message

    def check_static_motion(self, verbose=False):
        def get_variance(skeleton, joint_idx):
            wrist_pos = skeleton[:, joint_idx]
            variance = np.sum(np.var(wrist_pos, axis=0))
            return variance

        left_arm_var = get_variance(self.skeletons, 6)
        right_arm_var = get_variance(self.skeletons, 9)

        th = 0.002
        ret = left_arm_var < th and right_arm_var < th
        if verbose:
            print('check_static_motion: {}, left var {}, right var {}'.format(ret, left_arm_var, right_arm_var))
        return ret

    def check_frame_diff(self, verbose=False):
        diff = np.max(np.abs(np.diff(self.skeletons, axis=0, n=1)))

        th = 0.2
        ret = diff > th
        if verbose:
            print('check_frame_diff: {}, {:.5f}'.format(ret, diff))
        return ret

    def check_spine_angle(self, verbose=False):
        def angle_between(v1, v2):
            v1_u = v1 / np.linalg.norm(v1)
            v2_u = v2 / np.linalg.norm(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

        angles = []
        for i in range(self.skeletons.shape[0]):
            spine_vec = self.skeletons[i, 1] - self.skeletons[i, 0]
            angle = angle_between(spine_vec, [0, -1, 0])
            angles.append(angle)

        if np.rad2deg(max(angles)) > 30 or np.rad2deg(np.mean(angles)) > 20:
            if verbose:
                print('skip - check_spine_angle {:.5f}, {:.5f}'.format(max(angles), np.mean(angles)))
            return True
        else:
            if verbose:
                print('pass - check_spine_angle {:.5f}'.format(max(angles)))
            return False


