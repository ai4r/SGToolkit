import pprint
import time
import sys

[sys.path.append(i) for i in ['.', '..']]

import matplotlib
from torch import optim

from model.embedding_net import EmbeddingNet
import train_eval.train_gan
from utils.average_meter import AverageMeter
from utils.data_utils import convert_dir_vec_to_pose, convert_dir_vec_to_pose_torch
from utils.vocab_utils import build_vocab

matplotlib.use('Agg')  # we don't use interactive GUI

from config.parse_args import parse_args
from model.embedding_space_evaluator import EmbeddingSpaceEvaluator
from model.multimodal_context_net import PoseGenerator, ConvDiscriminator

from data_loader.lmdb_data_loader import *
import utils.train_utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def init_model(args, lang_model, pose_dim, _device):
    generator = discriminator = None
    if args.model == 'multimodal_context':  # ours
        generator = PoseGenerator(args,
                                  n_words=lang_model.n_words,
                                  word_embed_size=args.wordembed_dim,
                                  word_embeddings=lang_model.word_embedding_weights,
                                  pose_dim=pose_dim).to(_device)
        discriminator = ConvDiscriminator(pose_dim).to(_device)
    elif args.model == 'gesture_autoencoder':
        generator = EmbeddingNet(args, pose_dim, args.n_poses).to(_device)

    return generator, discriminator


def train_epochs(args, train_data_loader, test_data_loader, lang_model, pose_dim):
    start = time.time()
    loss_meters = [AverageMeter('loss'), AverageMeter('var_loss'), AverageMeter('gen'), AverageMeter('dis'),
                   AverageMeter('KLD'), AverageMeter('DIV_REG')]
    best_val_loss = (1e+10, 0)  # value, epoch

    # interval params
    print_interval = int(len(train_data_loader) / 5)
    save_sample_result_epoch_interval = 10
    save_model_epoch_interval = 20

    # init model
    generator, discriminator = init_model(args, lang_model, pose_dim, device)

    # use multi GPUs
    if torch.cuda.device_count() > 1:
        generator = torch.nn.DataParallel(generator)
        if discriminator is not None:
            discriminator = torch.nn.DataParallel(discriminator)

    # prepare an evaluator for FGD
    embed_space_evaluator = None
    if args.eval_net_path and len(args.eval_net_path) > 0:
        embed_space_evaluator = EmbeddingSpaceEvaluator(args, args.eval_net_path, lang_model, device)

    # define optimizers
    gen_optimizer = optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    dis_optimizer = None
    if discriminator is not None:
        dis_optimizer = torch.optim.Adam(discriminator.parameters(),
                                         lr=args.learning_rate * args.discriminator_lr_weight,
                                         betas=(0.5, 0.999))

    # training
    global_iter = 0
    best_values = {}  # best values for all loss metrics
    for epoch in range(args.epochs):
        # evaluate the test set
        val_metrics = evaluate_testset(test_data_loader, generator, embed_space_evaluator, args)

        for key in val_metrics.keys():
            if key not in best_values.keys() or val_metrics[key] < best_values[key][0]:
                best_values[key] = (val_metrics[key], epoch)

        # best?
        if 'FGD' in val_metrics.keys():
            val_loss = val_metrics['FGD']
        else:
            val_loss = val_metrics['loss']
        is_best = val_loss < best_val_loss[0]
        if is_best:
            logging.info('  *** BEST VALIDATION LOSS: {:.3f}'.format(val_loss))
            best_val_loss = (val_loss, epoch)
        else:
            logging.info('  best validation loss so far: {:.3f} at EPOCH {}'.format(best_val_loss[0], best_val_loss[1]))

        # save model
        if is_best or (epoch % save_model_epoch_interval == 0 and epoch > 0):
            dis_state_dict = None
            try:  # multi gpu
                gen_state_dict = generator.module.state_dict()
                if discriminator is not None:
                    dis_state_dict = discriminator.module.state_dict()
            except AttributeError:  # single gpu
                gen_state_dict = generator.state_dict()
                if discriminator is not None:
                    dis_state_dict = discriminator.state_dict()

            if is_best:
                save_name = '{}/{}_checkpoint_best.bin'.format(args.model_save_path, args.name)
            else:
                save_name = '{}/{}_checkpoint_{:03d}.bin'.format(args.model_save_path, args.name, epoch)

            utils.train_utils.save_checkpoint({
                'args': args, 'epoch': epoch, 'lang_model': lang_model,
                'pose_dim': pose_dim, 'gen_dict': gen_state_dict,
                'dis_dict': dis_state_dict,
            }, save_name)

        # save sample results
        if args.save_result_video and epoch % save_sample_result_epoch_interval == 0:
            evaluate_sample_and_save_video(
                epoch, args.name, test_data_loader, generator,
                args=args, lang_model=lang_model)

        # train iter
        iter_start_time = time.time()
        for iter_idx, data in enumerate(train_data_loader, 0):
            global_iter += 1
            in_text, text_lengths, in_text_padded, target_pose, target_vec, in_audio, style_vec, aux_info = data
            batch_size = target_vec.size(0)

            in_text_padded = in_text_padded.to(device)
            in_audio = in_audio.to(device)
            target_vec = target_vec.to(device)
            style_vec = style_vec.to(device)

            # train
            if args.model == 'multimodal_context':
                loss = train_eval.train_gan.train_iter_gan(
                    args, epoch, in_text_padded, in_audio, target_vec, style_vec,
                    generator, discriminator, gen_optimizer, dis_optimizer)
            else:
                assert False

            # loss values
            for loss_meter in loss_meters:
                name = loss_meter.name
                if name in loss:
                    loss_meter.update(loss[name], batch_size)

            # print training status
            if (iter_idx + 1) % print_interval == 0:
                print_summary = 'EP {} ({:3d}) | {:>8s}, {:.0f} samples/s | '.format(
                    epoch, iter_idx + 1, utils.train_utils.time_since(start),
                    batch_size / (time.time() - iter_start_time))
                for loss_meter in loss_meters:
                    if loss_meter.count > 0:
                        print_summary += '{}: {:.3f}, '.format(loss_meter.name, loss_meter.avg)
                        loss_meter.reset()
                logging.info(print_summary)

            iter_start_time = time.time()

    # print best losses
    logging.info('--------- best loss values ---------')
    for key in best_values.keys():
        logging.info('{}: {:.3f} at EPOCH {}'.format(key, best_values[key][0], best_values[key][1]))


def evaluate_testset(test_data_loader, generator, embed_space_evaluator, args):
    generator.train(False)  # to evaluation mode

    if embed_space_evaluator:
        embed_space_evaluator.reset()

    control_mode = ['none', 'pose', 'style']
    losses = AverageMeter('loss')
    joint_mae = AverageMeter('mae_on_joint')
    pose_compliance = AverageMeter('pose_compliance')
    style_compliance = AverageMeter('style_compliance')

    mean_vec = np.array(args.mean_dir_vec).squeeze()

    start = time.time()
    for mode in control_mode:
        for iter_idx, data in enumerate(test_data_loader, 0):
            in_text, text_lengths, in_text_padded, target_pose, target_vec, in_audio, style_vec, aux_info = data
            batch_size = target_vec.size(0)

            in_text = in_text.to(device)
            in_text_padded = in_text_padded.to(device)
            in_audio = in_audio.to(device)
            target = target_vec.to(device)
            style_vec = style_vec.to(device)

            # pose control
            pose_control = target.new_zeros((target.shape[0], target.shape[1], target.shape[2] + 1))
            pose_control[:, 0:args.n_pre_poses, :-1] = target[:, 0:args.n_pre_poses]
            pose_control[:, 0:args.n_pre_poses, -1] = 1  # mask bit to indicate positions being controlled
            if mode == 'pose':
                control_point = (args.n_pre_poses + pose_control.shape[1]) // 2
                pose_control[:, control_point, :-1] = target[:, control_point]
                pose_control[:, control_point, -1] = 1  # mask bit

            # style control
            if mode == 'style':
                pass
            else:
                style_vec = None  # no style input

            # inference
            with torch.no_grad():
                if args.model == 'multimodal_context':
                    out_dir_vec, *_ = generator(pose_control, in_text_padded, in_audio, style_vec)
                else:
                    assert False

            if args.model == 'multimodal_context':
                if mode == 'none':
                    loss = F.l1_loss(out_dir_vec, target)
                    losses.update(loss.item(), batch_size)

                    if embed_space_evaluator:
                        embed_space_evaluator.push_samples(out_dir_vec, target)

                    # calculate MAE of joint coordinates
                    out_dir_vec = out_dir_vec.cpu().numpy()
                    target_vec = target_vec.cpu().numpy()
                    if args.normalize_motion_data:
                        out_dir_vec += mean_vec
                        target_vec += mean_vec
                    out_joint_poses = convert_dir_vec_to_pose(out_dir_vec)
                    target_poses = convert_dir_vec_to_pose(target_vec)

                    if out_joint_poses.shape[1] == args.n_poses:
                        diff = out_joint_poses[:, args.n_pre_poses:] - target_poses[:, args.n_pre_poses:]
                    else:
                        diff = out_joint_poses - target_poses[:, args.n_pre_poses:]
                    joint_mae.update(np.mean(np.absolute(diff)), batch_size)
                elif mode == 'pose':
                    # calculate pose control compliance
                    pose_compliance_val = F.l1_loss(out_dir_vec[:, control_point], target[:, control_point]).item()
                    pose_compliance.update(pose_compliance_val, batch_size)
                elif mode == 'style':
                    # calculate style control compliance
                    mean_dir_vec = torch.as_tensor(args.mean_dir_vec).squeeze().to(out_dir_vec.device)
                    mean_pose = torch.as_tensor(args.mean_pose).squeeze().to(out_dir_vec.device)
                    style_stat = torch.tensor([args.style_val_mean, args.style_val_std, args.style_val_max]).squeeze().to(out_dir_vec.device)

                    if args.normalize_motion_data:
                        out_dir_vec += mean_dir_vec
                    out_joint_poses = convert_dir_vec_to_pose_torch(out_dir_vec)
                    window_size = args.motion_resampling_framerate * 2  # 2 sec

                    out_style = calculate_style_vec(out_joint_poses, window_size, mean_pose, style_stat)
                    style_compliance_val = F.l1_loss(out_style, style_vec).item()
                    style_compliance.update(style_compliance_val, batch_size)

    elapsed_time = time.time() - start
    generator.train(True)  # back to training mode

    # print
    ret_dict = {'loss': losses.avg, 'joint_mae': joint_mae.avg}
    if pose_compliance.count > 0:
        ret_dict['pose_compliance'] = pose_compliance.avg
    if style_compliance.count > 0:
        ret_dict['style_compliance'] = style_compliance.avg

    if embed_space_evaluator and embed_space_evaluator.get_no_of_samples() > 0:
        fgd, feat_dist = embed_space_evaluator.get_scores()
        ret_dict['FGD'] = fgd
        ret_dict['feat_dist'] = feat_dist

    log_str = '[VAL] '
    for k in ret_dict:
        log_str += f'{k}: {ret_dict[k]:.5f}, '
    log_str += f'[{elapsed_time:.1f}s]'
    logging.info(log_str)

    return ret_dict


def evaluate_sample_and_save_video(epoch, prefix, test_data_loader, generator, args, lang_model,
                                   n_save=None, save_path=None, use_pose_constraint=False, style_value=None):
    generator.train(False)  # eval mode
    if not n_save:
        n_save = 1 if epoch <= 0 else 5

    if use_pose_constraint:
        prefix = prefix + '_with_constraints'
    if style_value:
        prefix = prefix + '_style_{}'.format(style_value)

    out_raw = []

    mean_dir_vec = torch.tensor(args.mean_dir_vec).squeeze().to(device)

    with torch.no_grad():
        for iter_idx, data in enumerate(test_data_loader, 0):
            if iter_idx >= n_save:  # save N samples
                break

            in_text, text_lengths, in_text_padded, target_pose, target_dir_vec, in_audio, style_vec, aux_info = data

            # prepare
            select_index = 0
            in_text_padded = in_text_padded[select_index, :].unsqueeze(0).to(device)
            in_audio = in_audio[select_index, :].unsqueeze(0).to(device)
            target_dir_vec = target_dir_vec[select_index, :, :].unsqueeze(0).to(device)
            style_vec = style_vec[select_index].unsqueeze(0).to(device)

            input_words = []
            for i in range(in_text_padded.shape[1]):
                word_idx = int(in_text_padded.data[select_index, i])
                if word_idx > 0:
                    input_words.append(lang_model.index2word[word_idx])
            sentence = ' '.join(input_words)

            # style vector
            if style_value:
                style_vector = torch.FloatTensor(style_value).to(device)
                style_vector = style_vector.repeat(1, target_dir_vec.shape[1], 1)
            else:
                style_vector = style_vec

            # aux info
            aux_str = '({}, time: {}-{})'.format(
                aux_info['vid'][select_index],
                str(datetime.timedelta(seconds=aux_info['start_time'][select_index].item())),
                str(datetime.timedelta(seconds=aux_info['end_time'][select_index].item())))

            # inference
            pose_constraints = target_dir_vec.new_zeros((target_dir_vec.shape[0], target_dir_vec.shape[1],
                                                         target_dir_vec.shape[2] + 1))
            if not args.normalize_motion_data:
                # fill with mean data
                pose_constraints[:, :, :-1] = mean_dir_vec.repeat(target_dir_vec.shape[0], target_dir_vec.shape[1], 1)
            pose_constraints[:, 0:args.n_pre_poses, :-1] = target_dir_vec[:, 0:args.n_pre_poses]
            pose_constraints[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints

            if use_pose_constraint:
                n_samples = target_dir_vec.shape[0]
                copy_length = 5
                for i in range(n_samples):
                    copy_point = 50
                    pose_constraints[i, copy_point:copy_point + copy_length, :-1] = \
                        target_dir_vec[i, copy_point:copy_point + copy_length]
                    pose_constraints[i, copy_point:copy_point + copy_length, -1] = 1

            if args.model == 'multimodal_context':
                out_dir_vec, *_ = generator(pose_constraints, in_text_padded, in_audio, style_vector)

            # to video
            audio_npy = np.squeeze(in_audio.cpu().numpy())
            target_dir_vec = np.squeeze(target_dir_vec.cpu().numpy())
            out_dir_vec = np.squeeze(out_dir_vec.cpu().numpy())

            if save_path is None:
                save_path = args.model_save_path

            if args.normalize_motion_data:
                mean_data = np.array(args.mean_dir_vec).squeeze()
                target_dir_vec += mean_data
                out_dir_vec += mean_data

            utils.train_utils.create_video_and_save(
                save_path, epoch, prefix, iter_idx,
                target_dir_vec, out_dir_vec,
                sentence, audio=audio_npy, aux_str=aux_str)

            target_dir_vec = target_dir_vec.reshape((target_dir_vec.shape[0], 9, 3))
            out_dir_vec = out_dir_vec.reshape((out_dir_vec.shape[0], 9, 3))

            out_raw.append({
                'sentence': sentence,
                'audio': audio_npy,
                'human_dir_vec': target_dir_vec,
                'out_dir_vec': out_dir_vec,
                'aux_info': aux_str
            })

    generator.train(True)  # back to training mode

    return out_raw


def main(config):
    args = config['args']

    # random seed
    if args.random_seed >= 0:
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        os.environ['PYTHONHASHSEED'] = str(args.random_seed)

    # set logger
    utils.train_utils.set_logger(args.model_save_path, os.path.basename(__file__).replace('.py', '.log'))

    logging.info("PyTorch version: {}".format(torch.__version__))
    logging.info("CUDA version: {}".format(torch.version.cuda))
    logging.info("{} GPUs, default {}".format(torch.cuda.device_count(), device))
    logging.info(pprint.pformat(vars(args)))

    # build vocab
    vocab_cache_path = os.path.join(os.path.split(args.train_data_path)[0], 'vocab_cache.pkl')
    lang_model = build_vocab('words', [args.train_data_path, args.val_data_path, args.test_data_path],
                             vocab_cache_path, args.wordembed_path, args.wordembed_dim)

    # dataset
    collate_fn = default_collate_fn
    mean_dir_vec = np.array(args.mean_dir_vec).reshape(-1, 3)
    train_dataset = SpeechMotionDataset(args.train_data_path,
                                        n_poses=args.n_poses,
                                        subdivision_stride=args.subdivision_stride,
                                        pose_resampling_fps=args.motion_resampling_framerate,
                                        mean_dir_vec=mean_dir_vec,
                                        mean_pose=args.mean_pose,
                                        normalize_motion=args.normalize_motion_data,
                                        style_stat=[args.style_val_mean, args.style_val_std, args.style_val_max]
                                        )
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=args.loader_workers, pin_memory=True,
                              collate_fn=collate_fn
                              )

    val_dataset = SpeechMotionDataset(args.val_data_path,
                                      n_poses=args.n_poses,
                                      subdivision_stride=args.subdivision_stride,
                                      pose_resampling_fps=args.motion_resampling_framerate,
                                      mean_dir_vec=mean_dir_vec,
                                      mean_pose=args.mean_pose,
                                      normalize_motion=args.normalize_motion_data,
                                      style_stat=[args.style_val_mean, args.style_val_std, args.style_val_max]
                                      )
    test_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                             shuffle=False, drop_last=True, num_workers=args.loader_workers, pin_memory=True,
                             collate_fn=collate_fn
                             )

    train_dataset.set_lang_model(lang_model)
    val_dataset.set_lang_model(lang_model)

    # train
    pose_dim = 27  # 9 x 3
    train_epochs(args, train_loader, test_loader, lang_model, pose_dim=pose_dim)


if __name__ == '__main__':
    _args = parse_args()
    main({'args': _args})
