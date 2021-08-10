import datetime
import logging
import os
import pickle
import random

import librosa
import numpy as np
import lmdb as lmdb
import torch
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

import utils.train_utils
import utils.data_utils
from model.vocab import Vocab
from data_loader.data_preprocessor import DataPreprocessor
import pyarrow


def default_collate_fn(data):
    _, text_padded, pose_seq, vec_seq, audio, style_vec, aux_info = zip(*data)

    text_padded = default_collate(text_padded)
    pose_seq = default_collate(pose_seq)
    vec_seq = default_collate(vec_seq)
    audio = default_collate(audio)
    style_vec = default_collate(style_vec)
    aux_info = {key: default_collate([d[key] for d in aux_info]) for key in aux_info[0]}

    return torch.tensor([0]), torch.tensor([0]), text_padded, pose_seq, vec_seq, audio, style_vec, aux_info


def calculate_style_vec(pose_seq, window_size, mean_pose, style_mean_std=None):
    if pose_seq.shape[-1] != 3:
        pose_seq = pose_seq.reshape(pose_seq.shape[:-1] + (-1, 3))

    batch_size = pose_seq.shape[0]
    n_poses = pose_seq.shape[1]
    style_vec = torch.zeros((batch_size, n_poses, 3), dtype=pose_seq.dtype, device=pose_seq.device)
    half_window = window_size // 2

    for i in range(n_poses):
        start_idx = max(0, i - half_window)
        end_idx = min(n_poses, i + half_window)
        poses_roi = pose_seq[:, start_idx:end_idx]

        # motion speed
        diff = poses_roi[:, 1:] - poses_roi[:, :-1]
        motion_speed = torch.mean(torch.abs(diff), dim=(1, 2, 3))

        # motion acceleration
        # accel = diff[:, 1:] - diff[:, :-1]
        # motion_accel = torch.mean(torch.abs(accel), dim=(1, 2, 3))

        # space
        space = torch.norm(poses_roi[:, :, 6] - poses_roi[:, :, 9], dim=2)  # distance between two hands
        space = torch.mean(space, dim=1)

        # handedness
        left_arm_move = torch.mean(torch.abs(poses_roi[:, 1:, 6] - poses_roi[:, :-1, 6]), dim=(1, 2))
        right_arm_move = torch.mean(torch.abs(poses_roi[:, 1:, 9] - poses_roi[:, :-1, 9]), dim=(1, 2))

        handedness = torch.where(right_arm_move > left_arm_move,
                                 left_arm_move / right_arm_move - 1,  # (-1, 0]
                                 1 - right_arm_move / left_arm_move)  # [0, 1)
        handedness *= 3  # to [-3, 3]

        style_vec[:, i, 0] = motion_speed
        style_vec[:, i, 1] = space
        style_vec[:, i, 2] = handedness

    # normalize
    if style_mean_std is not None:
        mean, std, max_val = style_mean_std[0], style_mean_std[1], style_mean_std[2]
        style_vec = (style_vec - mean) / std
        style_vec = torch.clamp(style_vec, -3, 3)  # +-3std
        # style_vec = style_vec / max_val
        # style_vec = torch.clamp(style_vec, -1, 1)

    return style_vec


class SpeechMotionDataset(Dataset):
    def __init__(self, lmdb_dir, n_poses, subdivision_stride, pose_resampling_fps, mean_pose, mean_dir_vec,
                 normalize_motion=False, style_stat=None):
        self.lmdb_dir = lmdb_dir
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps

        self.expected_audio_length = int(round(n_poses / pose_resampling_fps * 16000))

        self.lang_model = None

        if mean_dir_vec.shape[-1] != 3:
            mean_dir_vec = mean_dir_vec.reshape(mean_dir_vec.shape[:-1] + (-1, 3))
        self.mean_dir_vec = mean_dir_vec
        self.normalize_motion = normalize_motion

        logging.info("Reading data '{}'...".format(lmdb_dir))
        preloaded_dir = lmdb_dir + '_cache'
        if not os.path.exists(preloaded_dir):
            data_sampler = DataPreprocessor(lmdb_dir, preloaded_dir, n_poses,
                                            subdivision_stride, pose_resampling_fps, mean_pose, mean_dir_vec)
            data_sampler.run()
        else:
            logging.info('Found pre-loaded samples from {}'.format(preloaded_dir))

        # init lmdb
        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()['entries']

        # pre-compute style vec
        precomputed_style = lmdb_dir + '_style_vec.npy'
        if not os.path.exists(precomputed_style):
            if style_stat is not None:
                logging.info('Calculating style vectors...')
                mean_pose = torch.tensor(mean_pose).squeeze()
                mean_dir_vec = torch.tensor(mean_dir_vec).squeeze()
                style_stat = torch.tensor(style_stat).squeeze()
                self.style_vectors = []
                with self.lmdb_env.begin(write=False) as txn:
                    for i in tqdm(range(self.n_samples)):
                        key = '{:010}'.format(i).encode('ascii')
                        sample = txn.get(key)
                        sample = pyarrow.deserialize(sample)
                        word_seq, pose_seq, vec_seq, audio, aux_info = sample

                        window_size = pose_resampling_fps * 2
                        poses = torch.from_numpy(vec_seq).unsqueeze(0)
                        if normalize_motion:
                            poses += mean_dir_vec  # unnormalize
                        poses = utils.data_utils.convert_dir_vec_to_pose_torch(poses)  # normalized bone lengths
                        style_vec = calculate_style_vec(poses, window_size, mean_pose, style_stat)
                        self.style_vectors.append(style_vec[0].numpy())
                self.style_vectors = np.stack(self.style_vectors)

                with open(precomputed_style, 'wb') as f:
                    np.save(f, self.style_vectors)
                print('style npy mean: ', np.mean(self.style_vectors, axis=(0, 1)))
                print('style npy std: ', np.std(self.style_vectors, axis=(0, 1)))
            else:
                self.style_vectors = None
        else:
            with open(precomputed_style, 'rb') as f:
                self.style_vectors = np.load(f)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = '{:010}'.format(idx).encode('ascii')
            sample = txn.get(key)

            sample = pyarrow.deserialize(sample)
            word_seq, pose_seq, vec_seq, audio, aux_info = sample

        def extend_word_seq(lang, words, end_time=None):
            n_frames = self.n_poses
            if end_time is None:
                end_time = aux_info['end_time']
            frame_duration = (end_time - aux_info['start_time']) / n_frames

            extended_word_indices = np.zeros(n_frames)  # zero is the index of padding token
            for word in words:
                idx = max(0, int(np.floor((word[1] - aux_info['start_time']) / frame_duration)))
                if idx < n_frames:
                    extended_word_indices[idx] = lang.get_word_index(word[0])
            return torch.Tensor(extended_word_indices).long()

        def words_to_tensor(lang, words, end_time=None):
            indexes = [lang.SOS_token]
            for word in words:
                if end_time is not None and word[1] > end_time:
                    break
                indexes.append(lang.get_word_index(word[0]))
            indexes.append(lang.EOS_token)
            return torch.Tensor(indexes).long()

        duration = aux_info['end_time'] - aux_info['start_time']
        if self.style_vectors is not None:
            style_vec = torch.from_numpy(self.style_vectors[idx])
        else:
            style_vec = torch.zeros((self.n_poses, 1))

        do_clipping = True
        if do_clipping:
            sample_end_time = aux_info['start_time'] + duration * self.n_poses / vec_seq.shape[0]
            audio = utils.data_utils.make_audio_fixed_length(audio, self.expected_audio_length)
            vec_seq = vec_seq[0:self.n_poses]
            pose_seq = pose_seq[0:self.n_poses]
            style_vec = style_vec[0:self.n_poses]
        else:
            sample_end_time = None

        # motion data normalization
        vec_seq = np.copy(vec_seq)
        if self.normalize_motion:
            vec_seq -= self.mean_dir_vec

        # to tensors
        word_seq_tensor = words_to_tensor(self.lang_model, word_seq, sample_end_time)
        extended_word_seq = extend_word_seq(self.lang_model, word_seq, sample_end_time)
        vec_seq = torch.as_tensor(vec_seq).reshape((vec_seq.shape[0], -1)).float()
        pose_seq = torch.as_tensor(np.copy(pose_seq)).reshape((pose_seq.shape[0], -1)).float()
        audio = torch.as_tensor(np.copy(audio)).float()
        style_vec = style_vec.float()

        return word_seq_tensor, extended_word_seq, pose_seq, vec_seq, audio, style_vec, aux_info

    def set_lang_model(self, lang_model):
        self.lang_model = lang_model

