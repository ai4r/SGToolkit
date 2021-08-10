import re
import math

import librosa
import numpy as np
import torch
from scipy.interpolate import interp1d
from sklearn.preprocessing import normalize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


skeleton_line_pairs = [(0, 1, 'b'), (1, 2, 'darkred'), (2, 3, 'r'), (3, 4, 'orange'), (1, 5, 'darkgreen'),
                       (5, 6, 'limegreen'), (6, 7, 'darkseagreen')]
dir_vec_pairs = [(0, 1, 0.26), (1, 2, 0.18), (2, 3, 0.14), (1, 4, 0.22), (4, 5, 0.36),
                 (5, 6, 0.33), (1, 7, 0.22), (7, 8, 0.36), (8, 9, 0.33)]  # adjacency and bone length


def normalize_string(s):
    """ lowercase, trim, and remove non-letter characters """
    s = s.lower().strip()
    s = re.sub(r"([,.!?])", r" \1 ", s)  # isolate some marks
    s = re.sub(r"(['])", r"", s)  # remove apostrophe
    s = re.sub(r"[^a-zA-Z0-9,.!?]+", r" ", s)  # replace other characters with whitespace
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def remove_tags_marks(text):
    reg_expr = re.compile('<.*?>|[.,:;!?]+')
    clean_text = re.sub(reg_expr, '', text)
    return clean_text


def extract_melspectrogram(y, sr=16000):
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, power=2)
    log_melspec = librosa.power_to_db(melspec, ref=np.max)  # mels x time
    log_melspec = log_melspec.astype('float16')
    return log_melspec


def calc_spectrogram_length_from_motion_length(n_frames, fps):
    ret = (n_frames / fps * 16000 - 1024) / 512 + 1
    return int(round(ret))


def resample_pose_seq(poses, duration_in_sec, fps):
    n = len(poses)
    x = np.arange(0, n)
    y = poses
    f = interp1d(x, y, axis=0, kind='linear', fill_value='extrapolate')
    expected_n = duration_in_sec * fps
    x_new = np.arange(0, n, n / expected_n)
    interpolated_y = f(x_new)
    if hasattr(poses, 'dtype'):
        interpolated_y = interpolated_y.astype(poses.dtype)
    return interpolated_y


def time_stretch_for_words(words, start_time, speech_speed_rate):
    for i in range(len(words)):
        if words[i][1] > start_time:
            words[i][1] = start_time + (words[i][1] - start_time) / speech_speed_rate
        words[i][2] = start_time + (words[i][2] - start_time) / speech_speed_rate

    return words


def make_audio_fixed_length(audio, expected_audio_length):
    n_padding = expected_audio_length - len(audio)
    if n_padding > 0:
        audio = np.pad(audio, (0, n_padding), mode='symmetric')
    else:
        audio = audio[0:expected_audio_length]
    return audio


def pose_pca_transform_npy(poses_npy, pca, out_torch=True):
    if len(poses_npy.shape) == 2:
        pca_poses = pca.transform(poses_npy).astype(np.float32)  # [N x D] -> [N x PCA_D]
    else:
        n_samples = poses_npy.shape[0]
        n_seq = poses_npy.shape[1]

        poses_npy = poses_npy.reshape((-1, poses_npy.shape[-1]))
        pca_poses = pca.transform(poses_npy).astype(np.float32)  # [N x D] -> [N x PCA_D]
        pca_poses = pca_poses.reshape((n_samples, n_seq, -1))

    if out_torch:
        return torch.from_numpy(pca_poses).to(device)
    else:
        return pca_poses


def pose_pca_transform(poses, pca):
    poses_npy = poses.data.cpu().numpy()
    return pose_pca_transform_npy(poses_npy, pca)


def pose_pca_inverse_transform_npy(pca_data_npy, pca, out_torch=True):
    if len(pca_data_npy.shape) == 2:  # (samples, dim)
        poses = pca.inverse_transform(pca_data_npy).astype(np.float32)  # [N x PCA_D] -> [N x D]
    else:  # (samples, seq, dim)
        n_samples = pca_data_npy.shape[0]
        n_seq = pca_data_npy.shape[1]

        pca_data_npy = pca_data_npy.reshape((-1, pca_data_npy.shape[-1]))
        poses = pca.inverse_transform(pca_data_npy).astype(np.float32)  # [N x PCA_D] -> [N x D]
        poses = poses.reshape((n_samples, n_seq, -1))

    if out_torch:
        return torch.from_numpy(poses).to(device)
    else:
        return poses


def pose_pca_inverse_transform(pca_data, pca):
    pca_data_npy = pca_data.data.cpu().numpy()
    return pose_pca_inverse_transform_npy(pca_data_npy, pca)


def convert_dir_vec_to_pose(vec):
    vec = np.array(vec)

    if vec.shape[-1] != 3:
        vec = vec.reshape(vec.shape[:-1] + (-1, 3))

    if len(vec.shape) == 2:
        joint_pos = np.zeros((10, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[pair[1]] = joint_pos[pair[0]] + pair[2] * vec[j]
    elif len(vec.shape) == 3:
        joint_pos = np.zeros((vec.shape[0], 10, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[:, pair[1]] = joint_pos[:, pair[0]] + pair[2] * vec[:, j]
    elif len(vec.shape) == 4:  # (batch, seq, 9, 3)
        joint_pos = np.zeros((vec.shape[0], vec.shape[1], 10, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[:, :, pair[1]] = joint_pos[:, :, pair[0]] + pair[2] * vec[:, :, j]
    else:
        assert False

    return joint_pos


def convert_dir_vec_to_pose_torch(vec):
    assert len(vec.shape) == 3 or (len(vec.shape) == 4 and vec.shape[-1] == 3)

    if vec.shape[-1] != 3:
        vec = vec.reshape(vec.shape[:-1] + (-1, 3))

    joint_pos = torch.zeros((vec.shape[0], vec.shape[1], 10, 3), dtype=vec.dtype, device=vec.device)
    for j, pair in enumerate(dir_vec_pairs):
        joint_pos[:, :, pair[1]] = joint_pos[:, :, pair[0]] + pair[2] * vec[:, :, j]

    return joint_pos


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


def convert_pose_seq_to_dir_vec(pose):
    if pose.shape[-1] != 3:
        pose = pose.reshape(pose.shape[:-1] + (-1, 3))

    if len(pose.shape) == 3:
        dir_vec = np.zeros((pose.shape[0], len(dir_vec_pairs), 3))
        for i, pair in enumerate(dir_vec_pairs):
            dir_vec[:, i] = pose[:, pair[1]] - pose[:, pair[0]]
            dir_vec[:, i, :] = normalize(dir_vec[:, i, :], axis=1)  # to unit length
    elif len(pose.shape) == 4:  # (batch, seq, ...)
        dir_vec = np.zeros((pose.shape[0], pose.shape[1], len(dir_vec_pairs), 3))
        for i, pair in enumerate(dir_vec_pairs):
            dir_vec[:, :, i] = pose[:, :, pair[1]] - pose[:, :, pair[0]]
        for j in range(dir_vec.shape[0]):  # batch
            for i in range(len(dir_vec_pairs)):
                dir_vec[j, :, i, :] = normalize(dir_vec[j, :, i, :], axis=1)  # to unit length
    else:
        assert False

    return dir_vec


def normalize_3d_pose(kps):
    line_pairs = [(1, 0, 'b'), (2, 1, 'b'), (3, 2, 'b'),
                  (4, 1, 'g'), (5, 4, 'g'), (6, 5, 'g'),
                  # left (https://github.com/kenkra/3d-pose-baseline-vmd/wiki/body)
                  (7, 1, 'r'), (8, 7, 'r'), (9, 8, 'r')]  # right

    def unit_vector(vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def angle_between(v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::

                >>> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >>> angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >>> angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        """
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def rotation_matrix(axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        axis = np.asarray(axis)
        axis = axis / math.sqrt(np.dot(axis, axis))
        a = math.cos(theta / 2.0)
        b, c, d = -axis * math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    n_frames = kps.shape[0]
    for i in range(n_frames):
        # refine spine angles
        spine_vec = kps[i, 1] - kps[i, 0]
        angle = angle_between([0, -1, 0], spine_vec)
        th = np.deg2rad(10)
        if angle > th:
            angle = angle - th
            rot = rotation_matrix(np.cross([0, -1, 0], spine_vec), angle)
            kps[i] = np.matmul(kps[i], rot)

        # rotate
        shoulder_vec = kps[i, 7] - kps[i, 4]
        angle = np.pi - np.math.atan2(shoulder_vec[2], shoulder_vec[0])  # angles on XZ plane
        # if i == 0:
        #     print(angle, np.rad2deg(angle))
        if 180 > np.rad2deg(angle) > 20:
            angle = angle - np.deg2rad(20)
            rotate = True
        elif 180 < np.rad2deg(angle) < 340:
            angle = angle - np.deg2rad(340)
            rotate = True
        else:
            rotate = False

        if rotate:
            rot = rotation_matrix([0, 1, 0], angle)
            kps[i] = np.matmul(kps[i], rot)

        # rotate 180 deg
        rot = rotation_matrix([0, 1, 0], np.pi)
        kps[i] = np.matmul(kps[i], rot)

        # size
        bone_lengths = []
        for pair in line_pairs:
            bone_lengths.append(np.linalg.norm(kps[i, pair[0], :] - kps[i, pair[1], :]))
        scale_factor = 0.2 / np.mean(bone_lengths)
        kps[i] *= scale_factor

    return kps
