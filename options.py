from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in

class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, "kitti_data"))
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(os.path.expanduser("~"), "tmp"))
        self.parser.add_argument("--split_root",
                                 type=str,
                                 default=None,
                                 help="optional root directory containing split folders (default: <repo>/splits)")

        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="monovit_mdp")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "odom", "benchmark", "endovis", "hamlyn", "c3vd"],
                                 default="eigen_zhou")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="endovis",
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test", "endovis", "hamlyn", "c3vd"])
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=256)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=320)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=150.0)
        self.parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])
        self.parser.add_argument("--c3vd_intrinsics_file",
                                 type=str,
                                 default=None,
                                 help="optional path to a C3VD intrinsics file; fixed normalized K is used if missing")

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=12)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)

        # ABLATION options
        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--predictive_mask",
                                 help="if set, uses a predictive masking scheme as in Zhou et al",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])
        self.parser.add_argument("--pose_model_type",
                                 type=str,
                                 help="normal or shared",
                                 default="separate_resnet",
                                 choices=["posecnn", "separate_resnet", "shared"])

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)
        self.parser.add_argument("--wandb_project",
                                 type=str,
                                 default="Monodepth2_monovit",
                                 help="Weights & Biases project name")
        self.parser.add_argument("--wandb_entity",
                                 type=str,
                                 default=None,
                                 help="optional W&B entity/team")
        self.parser.add_argument("--wandb_name",
                                 type=str,
                                 default=None,
                                 help="optional W&B run name")
        self.parser.add_argument("--wandb_mode",
                                 type=str,
                                 default="online",
                                 choices=["online", "offline", "disabled"],
                                 help="W&B mode")
        self.parser.add_argument("--wandb_tags",
                                 nargs="+",
                                 type=str,
                                 default=[],
                                 help="optional W&B tags")

        # EVALUATION options
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 type=float,
                                 help="if set multiplies predictions by this number",
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_filelist",
                                 type=str,
                                 default=None,
                                 help="optional path to a custom eval file list")
        self.parser.add_argument("--gt_depths_path",
                                 type=str,
                                 default=None,
                                 help="optional path to a custom gt_depths.npz")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="SERV-CT",
                                 choices=["eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10", "endovis", "hamlyn", "SERV-CT", "c3vd"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set, assume we are loading eigen results from npy but want to evaluate with KITTI benchmark crop.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 type=str,
                                 help="if set will output the disparities to this folder")
        self.parser.add_argument("--post_process",
                                 help="if set will perform flipping post-processing as in Monodepth v1",
                                 action="store_true")
        self.parser.add_argument("--c3vd_eval_min_depth",
                                 type=float,
                                 default=0.1,
                                 help="minimum depth for C3VD evaluation mask (mm)")
        self.parser.add_argument("--c3vd_eval_max_depth",
                                 type=float,
                                 default=100.0,
                                 help="maximum depth for C3VD evaluation mask (mm)")
        # EXTRA: TRAIN-TIME EVALUATION
        self.parser.add_argument("--eval_each_epoch",
                                 help="if set, runs evaluation at the end of every epoch (if ground truth available)",
                                 action="store_true")
        self.parser.add_argument("--eval_batch_size",
                                 type=int,
                                 help="batch size used for evaluation",
                                 default=16)
        self.parser.add_argument("--eval_weights_subfolder",
                                 type=str,
                                 help="relative subfolder under log_dir/model_name where weights are saved",
                                 default="models/weights")
        # EXTRA: HAMLYN STRICT NEIGHBORS
        self.parser.add_argument("--hamlyn_strict_neighbors",
                                 help="if set, for Hamlyn dataset use nearest neighbor frame when a frame is missing",
                                 action="store_true")
        self.parser.add_argument("--neighbor_search_max",
                                 type=int,
                                 help="max +/- frame index to search for neighbor when strict neighbor mode is enabled",
                                 default=10)

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
