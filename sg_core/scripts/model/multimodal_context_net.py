import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from model import vocab
import model.embedding_net
from model.tcn import TemporalConvNet


class AudioFeatExtractor(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.encoder = models.resnet18(pretrained=False)
        num_ftrs = self.encoder.fc.in_features
        self.encoder.fc = nn.Linear(num_ftrs, feat_dim)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # add channel dim
            x = x.repeat(1, 3, 1, 1)  # make 3-channels
        x = x.float()
        out = self.encoder(x)
        return out


class AudioEncoder(nn.Module):
    def __init__(self, n_frames, feat_dim=32):
        super().__init__()
        self.n_frames = n_frames
        self.feat_extractor = AudioFeatExtractor(feat_dim)

    def forward(self, spectrogram):
        # divide into blocks and extract features
        feat_list = []
        spectrogram_length = spectrogram.shape[2]
        block_start_pts = np.array(range(0, self.n_frames)) * spectrogram_length / self.n_frames
        for i in range(self.n_frames):
            if i-2 < 0:
                start = 0
            else:
                start = np.round(block_start_pts[i-2])

            if i+1 >= self.n_frames:
                end = spectrogram_length
            else:
                end = block_start_pts[i+1]

            start = int(np.floor(start))
            end = int(min(spectrogram_length, np.ceil(end)))
            spectrogram_roi = spectrogram[:, :, start:end]
            feat = self.feat_extractor(spectrogram_roi)
            feat_list.append(feat)

        out = torch.stack(feat_list, dim=1)
        return out


class WavEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_extractor = nn.Sequential(
            nn.Conv1d(1, 16, 15, stride=5, padding=1600),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(16, 32, 15, stride=6),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(32, 64, 15, stride=6),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(64, 32, 15, stride=6),
            # nn.BatchNorm1d(128),
            # nn.LeakyReLU(0.3, inplace=True),
            # nn.Conv2d(32, 32, (5, 1), padding=0, stride=1)
        )

    def forward(self, wav_data):
        wav_data = wav_data.unsqueeze(1)  # add channel dim
        out = self.feat_extractor(wav_data)
        return out.transpose(1, 2)  # to (batch x seq x dim)


class TextEncoderTCN(nn.Module):
    """ based on https://github.com/locuslab/TCN/blob/master/TCN/word_cnn/model.py """
    def __init__(self, args, n_words, embed_size=300, pre_trained_embedding=None,
                 kernel_size=2, dropout=0.3, emb_dropout=0.1):
        super(TextEncoderTCN, self).__init__()

        if pre_trained_embedding is not None:  # use pre-trained embedding (fasttext)
            assert pre_trained_embedding.shape[0] == n_words
            assert pre_trained_embedding.shape[1] == embed_size
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pre_trained_embedding),
                                                          freeze=args.freeze_wordembed)
        else:
            self.embedding = nn.Embedding(n_words, embed_size)

        num_channels = [args.hidden_size] * args.n_layers
        self.tcn = TemporalConvNet(embed_size, num_channels, kernel_size, dropout=dropout)

        self.decoder = nn.Linear(num_channels[-1], 32)
        self.drop = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout
        self.init_weights()

    def init_weights(self):
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, input):
        emb = self.drop(self.embedding(input))
        y = self.tcn(emb.transpose(1, 2)).transpose(1, 2)
        y = self.decoder(y)
        return y.contiguous(), 0


class PoseGenerator(nn.Module):
    def __init__(self, args, pose_dim, n_words, word_embed_size, word_embeddings):
        super().__init__()
        self.pre_length = args.n_pre_poses
        self.gen_length = args.n_poses - args.n_pre_poses
        self.z_type = args.z_type
        self.input_context = args.input_context
        self.style_vec_size = len(args.style_val_mean)*2  # *2 for indicating bit

        if self.input_context == 'both':
            self.in_size = 32 + 32 + pose_dim + 1  # audio_feat + text_feat + last pose + constraint bit
        elif self.input_context == 'none':
            self.in_size = pose_dim + 1
        else:
            self.in_size = 32 + pose_dim + 1  # audio or text only

        self.audio_encoder = WavEncoder()
        self.text_encoder = TextEncoderTCN(args, n_words, word_embed_size, pre_trained_embedding=word_embeddings,
                                           dropout=args.dropout_prob)

        if self.z_type == 'style_vector':
            # self.z_size = 16 + self.style_vec_size
            self.z_size = self.style_vec_size
            self.in_size += self.z_size

        self.hidden_size = args.hidden_size
        self.gru = nn.GRU(self.in_size, hidden_size=self.hidden_size, num_layers=args.n_layers, batch_first=True,
                          bidirectional=True, dropout=args.dropout_prob)
        self.out = nn.Sequential(
            # nn.Linear(hidden_size, pose_dim)
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.LeakyReLU(True),
            nn.Linear(self.hidden_size//2, pose_dim)
        )

        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True

    def forward(self, pose_constraints, in_text, in_audio, style_vector=None):
        decoder_hidden = None
        if self.do_flatten_parameters:
            self.gru.flatten_parameters()

        text_feat_seq = audio_feat_seq = None
        if self.input_context != 'none':
            # audio
            audio_feat_seq = self.audio_encoder(in_audio)  # output (bs, n_frames, feat_size)

            # text
            text_feat_seq, _ = self.text_encoder(in_text)
            assert(audio_feat_seq.shape[1] == text_feat_seq.shape[1])

        # z vector
        z_mu = z_logvar = None
        if self.z_type == 'style_vector' or self.z_type == 'random':
            z_context = torch.randn(in_text.shape[0], 16, device=in_text.device)
        else:  # no z
            z_context = None

        # make an input
        if self.input_context == 'both':
            in_data = torch.cat((pose_constraints, audio_feat_seq, text_feat_seq), dim=2)
        elif self.input_context == 'audio':
            in_data = torch.cat((pose_constraints, audio_feat_seq), dim=2)
        elif self.input_context == 'text':
            in_data = torch.cat((pose_constraints, text_feat_seq), dim=2)
        else:
            assert False

        if self.z_type == 'style_vector':
            repeated_z = z_context.unsqueeze(1)
            repeated_z = repeated_z.repeat(1, in_data.shape[1], 1)
            if style_vector is None:
                style_vector = torch.zeros((in_data.shape[0], in_data.shape[1], self.style_vec_size),
                                           device=in_data.device, dtype=torch.float32)
            else:
                ones = torch.ones((in_data.shape[0], in_data.shape[1], self.style_vec_size//2),
                                  device=in_data.device, dtype=torch.float32)
                zeros = torch.zeros((in_data.shape[0], in_data.shape[1], self.style_vec_size//2),
                                    device=in_data.device, dtype=torch.float32)
                # style_vec_bit = torch.where(torch.isnan(style_vector), zeros, ones)
                style_vec_bit = torch.where(style_vector == 0, zeros, ones)
                style_vector[~style_vec_bit.bool()] = 0  # set masked elements to zeros
                style_vector = torch.cat((style_vector.float(), style_vec_bit), dim=2)

                # masking on frames having constraining poses
                constraint_mask = (pose_constraints[:, :, -1] == 1)
                style_vector[constraint_mask] = 0

            # in_data = torch.cat((in_data, repeated_z, style_vector), dim=2)
            in_data = torch.cat((in_data, style_vector), dim=2)
        elif z_context is not None:
            repeated_z = z_context.unsqueeze(1)
            repeated_z = repeated_z.repeat(1, in_data.shape[1], 1)
            in_data = torch.cat((in_data, repeated_z), dim=2)

        # forward
        output, decoder_hidden = self.gru(in_data, decoder_hidden)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  # sum bidirectional outputs
        output = self.out(output.reshape(-1, output.shape[2]))
        decoder_outputs = output.reshape(in_data.shape[0], in_data.shape[1], -1)
        # decoder_outputs = torch.tanh(decoder_outputs)

        return decoder_outputs, z_context, z_mu, z_logvar


class Discriminator(nn.Module):
    def __init__(self, args, input_size, n_words=None, word_embed_size=None, word_embeddings=None):
        super().__init__()
        self.input_size = input_size

        if n_words and word_embed_size:
            self.text_encoder = TextEncoderTCN(n_words, word_embed_size, word_embeddings)
            input_size += 32
        else:
            self.text_encoder = None

        self.hidden_size = args.hidden_size
        self.gru = nn.GRU(input_size, hidden_size=self.hidden_size, num_layers=args.n_layers, bidirectional=True,
                          dropout=args.dropout_prob, batch_first=True)
        self.out = nn.Linear(self.hidden_size, 1)
        self.out2 = nn.Linear(args.n_poses, 1)

        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True

    def forward(self, poses, in_text=None):
        decoder_hidden = None
        if self.do_flatten_parameters:
            self.gru.flatten_parameters()

        # pose_diff = poses[:, 1:] - poses[:, :-1]

        if self.text_encoder:
            text_feat_seq, _ = self.text_encoder(in_text)
            poses = torch.cat((poses, text_feat_seq), dim=2)

        output, decoder_hidden = self.gru(poses, decoder_hidden)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  # sum bidirectional outputs

        # use the last N outputs
        batch_size = poses.shape[0]
        # output = output[:, -self.gen_length:]
        output = output.contiguous().view(-1, output.shape[2])
        output = self.out(output)  # apply linear to every output
        output = output.view(batch_size, -1)
        output = self.out2(output)
        output = torch.sigmoid(output)

        return output


class ConvDiscriminator(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

        self.hidden_size = 64
        self.pre_conv = nn.Sequential(
            nn.Conv1d(input_size, 16, 3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(True),
            nn.Conv1d(16, 8, 3),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(True),
            nn.Conv1d(8, 8, 3),
        )

        self.gru = nn.GRU(8, hidden_size=self.hidden_size, num_layers=4, bidirectional=True,
                          dropout=0.3, batch_first=True)
        self.out = nn.Linear(self.hidden_size, 1)
        self.out2 = nn.Linear(54, 1)

        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True

    def forward(self, poses, in_text=None):
        decoder_hidden = None
        if self.do_flatten_parameters:
            self.gru.flatten_parameters()

        poses = poses.transpose(1, 2)
        feat = self.pre_conv(poses)
        feat = feat.transpose(1, 2)

        output, decoder_hidden = self.gru(feat, decoder_hidden)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  # sum bidirectional outputs

        # use the last N outputs
        batch_size = poses.shape[0]
        # output = output[:, -self.gen_length:]
        output = output.contiguous().view(-1, output.shape[2])
        output = self.out(output)  # apply linear to every output
        output = output.view(batch_size, -1)
        output = self.out2(output)
        output = torch.sigmoid(output)

        return output

