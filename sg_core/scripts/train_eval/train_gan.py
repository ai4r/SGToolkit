import random

import numpy as np
import torch
import torch.nn.functional as F

import data_loader.lmdb_data_loader
from utils.data_utils import convert_dir_vec_to_pose_torch

from sg_core.scripts.train_eval.diff_augment import DiffAugment


def add_noise(data):
    noise = torch.randn_like(data) * 0.1
    return data + noise


def train_iter_gan(args, epoch, in_text, in_audio, target_data, style_vector,
                   pose_decoder, discriminator,
                   pose_dec_optim, dis_optim):
    warm_up_epochs = args.loss_warmup
    mean_dir_vec = torch.tensor(args.mean_dir_vec).squeeze().to(target_data.device)
    mean_pose = torch.tensor(args.mean_pose).squeeze().to(target_data.device)

    # make pose constraints
    pose_constraints = target_data.new_zeros((target_data.shape[0], target_data.shape[1], target_data.shape[2] + 1))
    if not args.normalize_motion_data:
        # fill with mean data
        pose_constraints[:, :, :-1] = mean_dir_vec.repeat(target_data.shape[0], target_data.shape[1], 1)
    pose_constraints[:, 0:args.n_pre_poses, :-1] = target_data[:, 0:args.n_pre_poses]
    pose_constraints[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
    if args.use_pose_control and random.random() < 0.5:
        n_samples = target_data.shape[0]

        copy_length = np.abs(np.random.triangular(-args.n_poses, 0, args.n_poses, n_samples).astype(np.int))
        copy_length = np.clip(copy_length, a_min=1, a_max=args.n_poses - args.n_pre_poses)

        for i in range(n_samples):
            copy_point = random.randint(args.n_pre_poses, args.n_poses - copy_length[i])
            pose_constraints[i, copy_point:copy_point + copy_length[i], :-1] = \
                target_data[i, copy_point:copy_point + copy_length[i]]
            pose_constraints[i, copy_point:copy_point + copy_length[i], -1] = 1

    if args.use_style_control and random.random() < 0.5:
        use_div_reg = True

        # random dropout style element
        n_drop = random.randint(0, 2)
        if n_drop > 0:
            drop_idxs = random.sample(range(style_vector.shape[-1]), k=n_drop)
            # style_vector[:, :, drop_idxs] = float('nan')
            style_vector[:, :, drop_idxs] = 0
    else:
        use_div_reg = False
        style_vector = None

    ###########################################################################################
    # train D
    dis_error = None
    if epoch > warm_up_epochs and args.loss_gan_weight > 0.0:
        dis_optim.zero_grad()

        out_dir_vec, *_ = pose_decoder(pose_constraints, in_text, in_audio,
                                       style_vector)  # out shape (batch x seq x dim)

        if args.diff_augment:
            dis_real = discriminator(DiffAugment(target_data), in_text)
            dis_fake = discriminator(DiffAugment(out_dir_vec.detach()), in_text)
        else:
            dis_real = discriminator(target_data, in_text)
            dis_fake = discriminator(out_dir_vec.detach(), in_text)

        dis_error = torch.sum(-torch.mean(torch.log(dis_real + 1e-8) + torch.log(1 - dis_fake + 1e-8)))  # ns-gan
        dis_error.backward()
        dis_optim.step()

    ###########################################################################################
    # train G
    pose_dec_optim.zero_grad()

    # decoding
    out_dir_vec, z, z_mu, z_logvar = pose_decoder(pose_constraints, in_text, in_audio, style_vector)

    # loss
    beta = 0.1
    l1_loss = F.smooth_l1_loss(out_dir_vec / beta, target_data / beta) * beta

    if args.diff_augment:
        dis_output = discriminator(DiffAugment(out_dir_vec), in_text)
    else:
        dis_output = discriminator(out_dir_vec, in_text)

    gen_error = -torch.mean(torch.log(dis_output + 1e-8))

    if args.z_type == 'style_vector' and use_div_reg and args.loss_reg_weight > 0.0:
        # calculate style control compliance
        style_stat = torch.tensor([args.style_val_mean, args.style_val_std, args.style_val_max]).squeeze().to(out_dir_vec.device)

        if args.normalize_motion_data:
            out_dir_vec += mean_dir_vec

        out_joint_poses = convert_dir_vec_to_pose_torch(out_dir_vec)
        window_size = args.motion_resampling_framerate * 2  # 2 sec

        out_style = data_loader.lmdb_data_loader.calculate_style_vec(out_joint_poses, window_size, mean_pose, style_stat)
        style_compliance = F.l1_loss(style_vector, out_style)

        loss = args.loss_l1_weight * l1_loss + args.loss_reg_weight * style_compliance
    else:
        loss = args.loss_l1_weight * l1_loss

    if epoch > warm_up_epochs:
        loss += args.loss_gan_weight * gen_error

    loss.backward()
    pose_dec_optim.step()

    ret_dict = {'loss': args.loss_l1_weight * l1_loss.item()}

    if epoch > warm_up_epochs and args.loss_gan_weight > 0.0:
        ret_dict['gen'] = args.loss_gan_weight * gen_error.item()
        ret_dict['dis'] = dis_error.item()

    return ret_dict
