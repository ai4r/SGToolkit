import json
import math
import pickle
import os
import logging
import random
import time

import soundfile as sf
import librosa
import torch
import torch.nn.functional as F
import numpy as np
import gentle

from data_loader.data_preprocessor import DataPreprocessor
from utils.data_utils import remove_tags_marks
from utils.train_utils import load_checkpoint_and_model
from utils.tts_helper import TTSHelper

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gentle_resources = gentle.Resources()


class GestureGenerator:
    def __init__(self, checkpoint_path, audio_cache_path=None):
        args, generator, lang_model, out_dim = load_checkpoint_and_model(
            checkpoint_path, device)
        self.args = args
        self.generator = generator
        self.lang_model = lang_model
        print(vars(args))

        if audio_cache_path is None:
            audio_cache_path = '../output/cached_wav'
        self.tts = TTSHelper(cache_path=audio_cache_path)

        # load mean vec
        self.mean_dir_vec = np.array(args.mean_dir_vec).flatten()
        self.mean_pose = np.array(args.mean_pose).flatten()

    @staticmethod
    def align_words(audio, text):
        # resample audio to 8K
        audio_8k = librosa.resample(audio, 16000, 8000)
        wave_file = 'temp.wav'
        sf.write(wave_file, audio_8k, 8000, 'PCM_16')

        # run gentle to align words
        aligner = gentle.ForcedAligner(gentle_resources, text, nthreads=2, disfluency=False,
                                       conservative=False)
        gentle_out = aligner.transcribe(wave_file, logging=logging)
        words_with_timestamps = []
        for gentle_word in gentle_out.words:
            if gentle_word.case == 'success':
                words_with_timestamps.append([gentle_word.word, gentle_word.start, gentle_word.end])

        return words_with_timestamps

    def generate(self, input_text, pose_constraints=None, style_values=None, voice=None):
        # voice
        voice_lower = str(voice).lower()
        if voice_lower == 'none' or voice_lower == 'female':
            voice_name = 'en-female_2'
        elif voice_lower == 'male':
            voice_name = 'en-male_2'
        else:
            voice_name = voice  # file path

        # make audio
        text_without_tags = remove_tags_marks(input_text)
        print(text_without_tags)

        if '.wav' in voice_name or '.mp3' in voice_name:  # use external audio file
            tts_filename = voice_name
            if not os.path.isfile(tts_filename):
                return None
        else:  # TTS
            tts_filename = self.tts.synthesis(input_text, voice_name=voice_name, verbose=True)

        audio, audio_sr = librosa.load(tts_filename, mono=True, sr=16000, res_type='kaiser_fast')

        # get timestamps (use caching)
        word_timestamps_cache = tts_filename.replace('.wav', '.json')
        if not os.path.exists(word_timestamps_cache):
            words_with_timestamps = self.align_words(audio, text_without_tags)
            with open(word_timestamps_cache, 'w') as outfile:
                json.dump(words_with_timestamps, outfile)
        else:
            with open(word_timestamps_cache) as json_file:
                words_with_timestamps = json.load(json_file)

        # run
        output = self.generate_core(audio, words_with_timestamps,
                                    pose_constraints=pose_constraints, style_value=style_values)

        # make output match to the audio length
        total_frames = math.ceil(len(audio) / 16000 * self.args.motion_resampling_framerate)
        output = output[:total_frames]

        return output, audio, tts_filename, words_with_timestamps

    def generate_core(self, audio, words, audio_sr=16000, pose_constraints=None, style_value=None, fade_out=False):
        args = self.args
        out_list = []
        n_frames = args.n_poses
        clip_length = len(audio) / audio_sr

        # pose constraints
        mean_vec = torch.from_numpy(np.array(args.mean_dir_vec).flatten())
        if pose_constraints is not None:
            assert pose_constraints.shape[1] == len(args.mean_dir_vec) + 1
            pose_constraints = torch.from_numpy(pose_constraints)
            mask = pose_constraints[:, -1] == 0
            if args.normalize_motion_data:  # make sure that un-constrained frames have zero or mean values
                pose_constraints[:, :-1] = pose_constraints[:, :-1] - mean_vec
                pose_constraints[mask, :-1] = 0
            else:
                pose_constraints[mask, :-1] = mean_vec
            pose_constraints = pose_constraints.unsqueeze(0).to(device)

        # divide into inference units and do inferences
        unit_time = args.n_poses / args.motion_resampling_framerate
        stride_time = (args.n_poses - args.n_pre_poses) / args.motion_resampling_framerate
        if clip_length < unit_time:
            num_subdivision = 1
        else:
            num_subdivision = math.ceil((clip_length - unit_time) / stride_time) + 1
        audio_sample_length = int(unit_time * audio_sr)
        end_padding_duration = 0

        print('{}, {}, {}, {}, {}'.format(num_subdivision, unit_time, clip_length, stride_time, audio_sample_length))

        out_dir_vec = None
        start = time.time()
        for i in range(0, num_subdivision):
            start_time = i * stride_time
            end_time = start_time + unit_time

            # prepare audio input
            audio_start = math.floor(start_time / clip_length * len(audio))
            audio_end = audio_start + audio_sample_length
            in_audio = audio[audio_start:audio_end]
            if len(in_audio) < audio_sample_length:
                if i == num_subdivision - 1:
                    end_padding_duration = audio_sample_length - len(in_audio)
                in_audio = np.pad(in_audio, (0, audio_sample_length - len(in_audio)), 'constant')
            in_audio = torch.from_numpy(in_audio).unsqueeze(0).to(device).float()

            # prepare text input
            word_seq = DataPreprocessor.get_words_in_time_range(word_list=words, start_time=start_time,
                                                                end_time=end_time)
            extended_word_indices = np.zeros(n_frames)  # zero is the index of padding token
            frame_duration = (end_time - start_time) / n_frames
            for word in word_seq:
                print(word[0], end=', ')
                idx = max(0, int(np.floor((word[1] - start_time) / frame_duration)))
                extended_word_indices[idx] = self.lang_model.get_word_index(word[0])
            print(' ')
            in_text_padded = torch.LongTensor(extended_word_indices).unsqueeze(0).to(device)

            # prepare pre constraints
            start_frame = (args.n_poses - args.n_pre_poses) * i
            end_frame = start_frame + args.n_poses

            if pose_constraints is None:
                in_pose_const = torch.zeros((1, n_frames, len(args.mean_dir_vec) + 1))
                if not args.normalize_motion_data:
                    in_pose_const[:, :, :-1] = mean_vec
            else:
                in_pose_const = pose_constraints[:, start_frame:end_frame, :]

                if in_pose_const.shape[1] < n_frames:
                    n_pad = n_frames - in_pose_const.shape[1]
                    in_pose_const = F.pad(in_pose_const, [0, 0, 0, n_pad, 0, 0], "constant", 0)

            if i > 0:
                in_pose_const[0, 0:args.n_pre_poses, :-1] = out_dir_vec.squeeze(0)[-args.n_pre_poses:]
                in_pose_const[0, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
            in_pose_const = in_pose_const.float().to(device)

            # style vector
            if style_value is None:
                style_vector = None
            elif isinstance(style_value, list) or len(style_value.shape) == 1:  # global style
                style_value = np.nan_to_num(style_value)  # nan to zero
                style_vector = torch.FloatTensor(style_value).to(device)
                style_vector = style_vector.repeat(1, in_text_padded.shape[1], 1)
            else:
                style_value = np.nan_to_num(style_value)  # nan to zero
                style_vector = style_value[start_frame:end_frame]
                n_pad = in_text_padded.shape[1] - style_vector.shape[0]
                if n_pad > 0:
                    style_vector = np.pad(style_vector, ((0, n_pad), (0, 0)), 'constant', constant_values=0)
                style_vector = torch.FloatTensor(style_vector).to(device).unsqueeze(0)

            # inference
            print(in_text_padded)
            out_dir_vec, *_ = self.generator(in_pose_const, in_text_padded, in_audio, style_vector)
            out_seq = out_dir_vec[0, :, :].data.cpu().numpy()

            # smoothing motion transition
            if len(out_list) > 0:
                last_poses = out_list[-1][-args.n_pre_poses:]
                out_list[-1] = out_list[-1][:-args.n_pre_poses]  # delete last {n_pre_poses} frames

                for j in range(len(last_poses)):
                    n = len(last_poses)
                    prev = last_poses[j]
                    next = out_seq[j]
                    out_seq[j] = prev * (n - j) / (n + 1) + next * (j + 1) / (n + 1)

            out_list.append(out_seq)

        print('Avg. inference time: {:.2} s'.format((time.time() - start) / num_subdivision))

        # aggregate results
        out_dir_vec = np.vstack(out_list)

        # fade out to the mean pose
        if fade_out:
            n_smooth = args.n_pre_poses
            start_frame = len(out_dir_vec) - int(end_padding_duration / audio_sr * args.motion_resampling_framerate)
            end_frame = start_frame + n_smooth * 2
            if len(out_dir_vec) < end_frame:
                out_dir_vec = np.pad(out_dir_vec, [(0, end_frame - len(out_dir_vec)), (0, 0)], mode='constant')

            # fade out to mean poses
            if args.normalize_motion_data:
                out_dir_vec[end_frame - n_smooth:] = np.zeros((len(args.mean_dir_vec)))
            else:
                out_dir_vec[end_frame - n_smooth:] = args.mean_dir_vec

            # interpolation
            y = out_dir_vec[start_frame:end_frame]
            x = np.array(range(0, y.shape[0]))
            w = np.ones(len(y))
            w[0] = 5
            w[-1] = 5
            coeffs = np.polyfit(x, y, 2, w=w)
            fit_functions = [np.poly1d(coeffs[:, k]) for k in range(0, y.shape[1])]
            interpolated_y = [fit_functions[k](x) for k in range(0, y.shape[1])]
            interpolated_y = np.transpose(np.asarray(interpolated_y))  # (num_frames x dims)

            out_dir_vec[start_frame:end_frame] = interpolated_y

        if args.normalize_motion_data:
            output = out_dir_vec + self.mean_dir_vec  # unnormalize
        else:
            output = out_dir_vec

        return output
