import configargparse


def str2bool(v):
    """ from https://stackoverflow.com/a/43357954/1361529 """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = configargparse.ArgParser()
    parser.add('-c', '--config', required=True, is_config_file=True, help='Config file path')
    parser.add("--name", type=str, default="main")
    parser.add("--train_data_path", type=str, required=True)
    parser.add("--val_data_path", type=str, required=True)
    parser.add("--test_data_path", type=str, required=False)
    parser.add("--model_save_path", required=True)
    parser.add("--pose_representation", type=str, default='pca')
    parser.add("--pose_norm_stats_path", type=str, default=None)
    parser.add("--pose_representation_path", type=str, default=None)
    parser.add("--mean_dir_vec", action="append", type=float, nargs='*')
    parser.add("--mean_pose", action="append", type=float, nargs='*')
    parser.add("--style_val_max", action="append", type=float, nargs='*')
    parser.add("--style_val_mean", action="append", type=float, nargs='*')
    parser.add("--style_val_std", action="append", type=float, nargs='*')
    parser.add("--random_seed", type=int, default=-1)
    parser.add("--save_result_video", type=str2bool, default=True)

    # word embedding
    parser.add("--wordembed_path", type=str, default=None)
    parser.add("--wordembed_dim", type=int, default=200)
    parser.add("--freeze_wordembed", type=str2bool, default=False)

    # model
    parser.add("--model", type=str, required=True)
    parser.add("--epochs", type=int, default=10)
    parser.add("--batch_size", type=int, default=50)
    parser.add("--dropout_prob", type=float, default=0.3)
    parser.add("--n_layers", type=int, default=2)
    parser.add("--hidden_size", type=int, default=200)
    parser.add("--residual_output", type=str2bool, default=False)
    parser.add("--z_type", type=str, default='none')
    parser.add("--input_context", type=str, default='both')  # text, audio, both
    parser.add("--use_pose_control", type=str2bool, default=True)
    parser.add("--use_style_control", type=str2bool, default=True)

    # dataset
    parser.add("--motion_resampling_framerate", type=int, default=24)
    parser.add("--n_poses", type=int, default=50)
    parser.add("--n_pre_poses", type=int, default=5)
    parser.add("--subdivision_stride", type=int, default=5)
    parser.add("--normalize_motion_data", type=str2bool, default=False)
    parser.add("--augment_data", type=str2bool, default=False)
    parser.add("--loader_workers", type=int, default=0)

    # GAN parameter
    parser.add("--GAN_noise_size", type=int, default=0)

    # training
    parser.add("--diff_augment", type=str2bool, default=True)
    parser.add("--learning_rate", type=float, default=0.001)
    parser.add("--discriminator_lr_weight", type=float, default=0.2)
    parser.add("--loss_l1_weight", type=float, default=50)
    parser.add("--loss_gan_weight", type=float, default=1.0)
    parser.add("--loss_reg_weight", type=float, default=0.01)
    parser.add("--loss_warmup", type=int, default=-1)

    # eval
    parser.add("--eval_net_path", type=str, default='')

    args = parser.parse_args()
    return args
