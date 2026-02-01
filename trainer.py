from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import json

from networks.mpvit import mpvit_small
from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
import wandb
import matplotlib.pyplot as plt
_DEPTH_COLORMAP = plt.get_cmap('plasma', 256)  # for plotting


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # Ensure input dimensions are multiples of 32 (required by encoder)
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        # For pose, use 2 frames if using pairs (monocular sequence) or all frames otherwise
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"
        # Use pose net unless using stereo with only frame 0
        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            # Add stereo frame id if using stereo training
            self.opt.frame_ids.append("s")

        # **Depth Encoder**: MPViT (MonoViT) backbone
        self.models["encoder"] = mpvit_small()  # default in_chans=3 for RGB input
        # Manually set num_ch_enc based on MPViT output channels (including stem)
        self.models["encoder"].num_ch_enc = [64, 128, 216, 288, 288]
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        # Depth decoder
        self.models["depth"] = networks.DepthDecoder(self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        # **Pose Network** (for monocular training)
        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                # Separate pose encoder using MPViT (accepts concatenated pair of images as 6-channel input)
                pose_enc_channels = 3 * self.num_pose_frames  # e.g., 6 channels for two frames
                self.models["pose_encoder"] = mpvit_small(in_chans=pose_enc_channels)
                self.models["pose_encoder"].num_ch_enc = [64, 128, 216, 288, 288]
                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())
                # Pose decoder takes the pose encoder's feature channels; predict pose for 2 frames (target and one source)
                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2
                )
                self.models["pose"].to(self.device)
                self.parameters_to_train += list(self.models["pose"].parameters())

            elif self.opt.pose_model_type == "shared":
                # Shared encoder for pose (use the same features from depth encoder)
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc,
                    self.num_pose_frames  # e.g., 2 for pairs (predict 1 relative pose)
                )
                self.models["pose"].to(self.device)
                self.parameters_to_train += list(self.models["pose"].parameters())

            elif self.opt.pose_model_type == "posecnn":
                # PoseCNN uses raw image concatenation for pose (different architecture)
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2
                )
                self.models["pose"].to(self.device)
                self.parameters_to_train += list(self.models["pose"].parameters())

        # Predictive mask (if using the predictive masking baseline)
        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, "When using predictive_mask, please add --disable_automasking"
            # The predictive mask decoder has the same architecture as the depth decoder
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc,
                self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1)
            )
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        # Optimizer setup
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = None

        # Load pre-trained weights if specified
        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:", self.opt.model_name)
        print("Models and logs will be saved to:", self.opt.log_dir)
        print("Using device:", self.device)

        # Datasets
        datasets_dict = {
            "kitti": datasets.KITTIRAWDataset,
            "kitti_odom": datasets.KITTIOdomDataset,
            "endovis": datasets.SCAREDDataset,
            "hamlyn": datasets.HamlynDataset
        }
        self.dataset = datasets_dict[self.opt.dataset]

        splits_dir = os.path.join(os.path.dirname(__file__), "splits")
        fpath = os.path.join(splits_dir, self.opt.split, "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_path = fpath.format("val")
        test_path = fpath.format("test")
        # Hamlyn dataset might not have a separate val split; handle accordingly
        if os.path.exists(val_path):
            val_filenames = readlines(val_path)
        else:
            if os.path.exists(test_path):
                print(f"[split] WARNING: {val_path} not found. Using {test_path} as validation.")
                val_filenames = readlines(test_path)
            else:
                print(f"[split] WARNING: {val_path} and {test_path} not found. Splitting train into train/val.")
                all_train = train_filenames
                # Take 5% of training data as validation if no explicit split is provided
                n_val = max(1, int(0.05 * len(all_train)))
                val_filenames = all_train[:n_val]
                train_filenames = all_train[n_val:]

        img_ext = ".png" if self.opt.png else ".jpg"
        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        # Initialize datasets and dataloaders
        train_dataset = self.dataset(self.opt.data_path, train_filenames,
                                     self.opt.height, self.opt.width,
                                     self.opt.frame_ids, 4,
                                     is_train=True, img_ext=img_ext)
        val_dataset = self.dataset(self.opt.data_path, val_filenames,
                                   self.opt.height, self.opt.width,
                                   self.opt.frame_ids, 4,
                                   is_train=False, img_ext=img_ext)
        # If using Hamlyn, configure neighbor frame search parameters
        if self.opt.dataset == "hamlyn":
            train_dataset.strict_neighbors = self.opt.hamlyn_strict_neighbors
            train_dataset.neighbor_search_max = self.opt.neighbor_search_max
            val_dataset.strict_neighbors = self.opt.hamlyn_strict_neighbors
            val_dataset.neighbor_search_max = self.opt.neighbor_search_max

        self.train_loader = DataLoader(train_dataset, self.opt.batch_size, shuffle=True,
                                       num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(val_dataset, self.opt.batch_size, shuffle=False,
                                     num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        # Prepare projection and backprojection for each scale
        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)
            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w).to(self.device)
            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w).to(self.device)

        self.depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]
        print(f"Using split: {self.opt.split}")
        print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

        self.save_opts()

    def set_train(self):
        """Set all models to training mode."""
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Set all models to evaluation mode."""
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the full training pipeline."""
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            # Save weights at specified frequency
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                save_folder = self.save_model()
                # Optionally evaluate on test set after each save (if enabled and data available)
                if self.opt.eval_each_epoch:
                    eval_opt = copy.deepcopy(self.opt)
                    eval_opt.load_weights_folder = save_folder
                    eval_opt.eval_mono = True
                    eval_opt.eval_stereo = False
                    eval_opt.eval_split = "hamlyn"  # evaluate on Hamlyn test set
                    eval_opt.no_eval = False
                    try:
                        import evaluate_hr_depth
                        evaluate_hr_depth.evaluate(eval_opt)
                    except Exception as e:
                        print(f"Error during evaluation: {e}")
                    # Log evaluation metrics to Weights & Biases, if results.txt exists
                    try:
                        with open("results.txt", "r") as f:
                            lines = f.read().splitlines()
                        metrics_line = next((ln for ln in lines[::-1] if ln.strip().startswith("&")), None)
                        if metrics_line:
                            metrics_line = metrics_line.replace("\\\\", "")
                            parts = [p.strip() for p in metrics_line.split("&") if p.strip()]
                            if len(parts) >= 7:
                                vals = list(map(float, parts[:7]))
                                eval_metrics = {
                                    "de/abs_rel": vals[0], "de/sq_rel": vals[1],
                                    "de/rms": vals[2], "de/log_rms": vals[3],
                                    "da/a1": vals[4], "da/a2": vals[5], "da/a3": vals[6]
                                }
                                wandb.log(eval_metrics, step=self.step)
                                print(f"Epoch {self.epoch + 1} evaluation: {eval_metrics}")
                        # If results.txt not found, skip logging
                    except FileNotFoundError:
                        print("Warning: results.txt not found after evaluation; no metrics logged.")

    def run_epoch(self):
        """Run one epoch of training and validation."""
        print(f"Epoch {self.epoch + 1}/{self.opt.num_epochs} - Training")
        self.set_train()
        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time
            # Log at regular intervals
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = (self.step != 0) and (self.step % 2000 == 0)
            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().item())
                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)
                self.log("train", inputs, outputs, losses)
                self.val()  # run one validation step
            self.step += 1

        if self.model_lr_scheduler is not None:
            self.model_lr_scheduler.step()

    def process_batch(self, inputs):
        """Forward a minibatch through the network and compute losses."""
        # Move all items to device
        for key, ipt in list(inputs.items()):
            inputs[key] = ipt.to(self.device)

        # Predict depth for frame 0 (and other frames if shared encoder is used)
        if self.opt.pose_model_type == "shared":
            # Shared encoder: forward all frames together, then split features per frame
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids], 0)
            all_features = self.models["encoder"](all_color_aug)
            # Split each feature map along batch dimension into chunks for each frame
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]
            # `features` is a dict: frame_id -> list of feature maps for that frame
            features = {frame_id: [f[i] for f in all_features] for i, frame_id in enumerate(self.opt.frame_ids)}
            outputs = self.models["depth"](features[0])  # depth predictions using features of frame 0
        else:
            # Separate encoder (or no pose net): only use frame 0 for depth
            features = self.models["encoder"](inputs[("color_aug", 0, 0)])
            outputs = self.models["depth"](features)

        # If predictive masking is used, compute mask
        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        # If using pose net, predict poses for target frame 0 w.r.t. other frames
        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))
        # Generate reconstructed images for each scale/frame
        self.generate_images_pred(inputs, outputs)
        # Compute self-supervised losses
        losses = self.compute_losses(inputs, outputs)
        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between frame 0 and other frames for monocular training."""
        outputs = {}
        if self.num_pose_frames == 2:
            # Compute pose to each source frame (monocular two-frame case)
            for frame_id in self.opt.frame_ids[1:]:
                if frame_id == "s":  # skip stereo frames if present
                    continue

                # Use temporal ordering for input pair: if frame_id is negative, put it first
                if frame_id < 0:
                    img1 = inputs[("color", frame_id, 0)]
                    img2 = inputs[("color", 0, 0)]
                else:
                    img1 = inputs[("color", 0, 0)]
                    img2 = inputs[("color", frame_id, 0)]

                if self.opt.pose_model_type == "separate_resnet":
                    # Use separate pose encoder with concatenated images
                    pose_input = torch.cat([img1, img2], dim=1)  # (B, 6, H, W)
                    features_pair = self.models["pose_encoder"](pose_input)
                    axisangle, translation = self.models["pose"]([features_pair])
                elif self.opt.pose_model_type == "posecnn":
                    # Use PoseCNN directly on concatenated images
                    pose_input = torch.cat([img1, img2], dim=1)
                    axisangle, translation = self.models["pose"](pose_input)
                else:
                    # Shared encoder: use precomputed depth features for frame 0 and the other frame
                    axisangle, translation = self.models["pose"]([features[0], features[frame_id]])

                outputs[("axisangle", 0, frame_id)] = axisangle
                outputs[("translation", 0, frame_id)] = translation
                # Convert axisangle + translation to transformation matrix (camera 0 -> camera frame_id)
                outputs[("cam_T_cam", 0, frame_id)] = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=(frame_id < 0)
                )
        else:
            # Multi-frame pose prediction (if implemented)
            if self.opt.pose_model_type == "posecnn":
                # PoseCNN with all frames concatenated
                concat_imgs = torch.cat([inputs[("color", i, 0)] for i in self.opt.frame_ids], 1)
                axisangle, translation = self.models["pose"](concat_imgs)
                outputs[("axisangle", 0, 1)] = axisangle
                outputs[("translation", 0, 1)] = translation
            else:
                raise NotImplementedError("Multi-frame pose prediction not implemented")
        return outputs

    def val(self):
        """Validate the model on a single minibatch from the validation set."""
        self.set_eval()
        try:
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)
        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)
            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)
            self.log("val", inputs, outputs, losses)
        self.set_train()

    def log_time(self, batch_idx, duration, loss):
        """Log timing and loss to the console."""
        samples_per_sec = self.opt.batch_size / duration
        print(f"Epoch {self.epoch + 1} | Batch {batch_idx} | {samples_per_sec:.2f} samples/s | Loss: {loss:.5f}")

    def log(self, mode, inputs, outputs, losses):
        """
        Log a minibatch of results to Weights & Biases.
        `mode` is "train" or "val".
        """
        log_data = {}
        viz_idx = 0  # visualize first element of batch
        for s in self.opt.scales:
            # Log the target image at scale s
            tgt_img = inputs[("color", 0, s)][viz_idx].detach().cpu()
            log_data[f"{mode}_color/0_{s}"] = wandb.Image(tgt_img)
            # Log source images and their reconstructions
            for frame_id in self.opt.frame_ids:
                if frame_id == 0:
                    continue
                src_img = inputs[("color", frame_id, s)][viz_idx].detach().cpu()
                log_data[f"{mode}_color/{frame_id}_{s}"] = wandb.Image(src_img)
                if ("color", frame_id, s) in outputs:
                    pred_img = outputs[("color", frame_id, s)][viz_idx].detach().cpu()
                    log_data[f"{mode}_color_pred/{frame_id}_{s}"] = wandb.Image(pred_img)
            # Log the disparity map (colored)
            disp = outputs[("disp", s)][viz_idx, 0].detach().cpu().numpy()
            disp_color = self.colormap(disp)  # returns (C, H, W) array
            disp_color = disp_color.transpose(1, 2, 0)  # to (H, W, C)
            log_data[f"{mode}_disp/{s}"] = wandb.Image(disp_color)
        # Log scalar losses/metrics
        for key, val in losses.items():
            # Replace '/' with '_' for logging keys
            metric_name = key.replace("/", "_")
            log_data[f"{mode}_{metric_name}"] = val if isinstance(val, float) else val.cpu().item()
        wandb.log(log_data, step=self.step)

    def save_opts(self):
        """Save training options to disk."""
        models_dir = os.path.join(self.log_path, "models")
        os.makedirs(models_dir, exist_ok=True)
        to_save = self.opt.__dict__.copy()
        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk and return the save folder path."""
        models_dir = os.path.join(self.log_path, self.opt.eval_weights_subfolder)
        os.makedirs(models_dir, exist_ok=True)
        save_folder = os.path.join(models_dir, f"epoch_{self.epoch}")
        os.makedirs(save_folder, exist_ok=True)
        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, f"{model_name}.pth")
            state_dict = model.state_dict()
            if model_name == "encoder":
                # Store image size and stereo flag with encoder weights for loading
                state_dict["height"] = self.opt.height
                state_dict["width"] = self.opt.width
                state_dict["use_stereo"] = self.opt.use_stereo
            torch.save(state_dict, save_path)
        # Save optimizer state
        torch.save(self.model_optimizer.state_dict(), os.path.join(save_folder, "adam.pth"))
        print(f"Model saved to {save_folder}")
        return save_folder

    def load_model(self):
        """Load model(s) from a given folder path."""
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)
        assert os.path.isdir(self.opt.load_weights_folder), f"Cannot find folder {self.opt.load_weights_folder}"
        print(f"Loading model from folder {self.opt.load_weights_folder}")
        for n in self.opt.models_to_load:
            print(f"Loading {n} weights...")
            model_path = os.path.join(self.opt.load_weights_folder, f"{n}.pth")
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(model_path)
            # Filter out keys that are not in current model (e.g., different shapes)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)
        # Load optimizer state if exists
        opt_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(opt_path):
            print("Loading Adam optimizer state")
            optimizer_dict = torch.load(opt_path)
            self.model_optimizer.load_state_dict(optimizer_dict)

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for each source frame at each scale."""
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                # Upsample predicted disparity to input size for computing depth
                disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                if frame_id == "s":
                    # Stereo transformation already provided in inputs
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]
                if self.opt.pose_model_type == "posecnn":
                    # For PoseCNN, adjust transformation using mean inverse depth
                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]
                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)
                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0
                    )
                # Project depth from frame 0 into the source frame to sample pixels
                cam_points = self.backproject_depth[source_scale](depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](cam_points, inputs[("K", source_scale)], T)
                outputs[("sample", frame_id, scale)] = pix_coords
                # Sample colors from source image
                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)], outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True
                )
                if not self.opt.disable_automasking:
                    # Also store the unmodified source image for identity reprojection loss
                    outputs[("color_identity", frame_id, scale)] = inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Compute reprojection (photometric) loss between a predicted image and its target image."""
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute total loss as sum of reprojection and smoothness losses over all scales."""
        losses = {}
        total_loss = 0
        for scale in self.opt.scales:
            loss = 0.0
            reprojection_losses = []
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0
            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]
            # Photometric reprojection loss for each source frame
            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)
            if not self.opt.disable_automasking:
                # Compute identity reprojection loss (comparing target with the source images themselves)
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(self.compute_reprojection_loss(pred, target))
                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
                # Optionally average losses instead of stacking (for v1 multiscale)
                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    identity_reprojection_loss = identity_reprojection_losses
            elif self.opt.predictive_mask:
                # If using predictive mask, apply it to reprojection losses
                mask = outputs[("predictive_mask", scale)]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(mask, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                reprojection_losses *= mask
                # Small regularization to encourage mask to be all ones
                weighting_loss = 0.2 * torch.nn.BCELoss()(mask, torch.ones_like(mask))
                loss += weighting_loss.mean()
            # Take per-pixel minimum reprojection loss
            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses
            if not self.opt.disable_automasking:
                # To break ties, add a small random noise to the identity loss
                identity_reprojection_loss += torch.randn_like(identity_reprojection_loss) * 1e-5
                # Stack identity and reprojection losses
                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss
            # If we have stacked losses, choose the minimum per pixel
            if combined.shape[1] == 1:
                to_optimize = combined
            else:
                to_optimize, idxs = torch.min(combined, dim=1)
            if not self.opt.disable_automasking:
                # Store which pixels chose the identity loss (for debugging/visualization)
                outputs[f"identity_selection/{scale}"] = (idxs > identity_reprojection_loss.shape[1] - 1).float()
            loss += to_optimize.mean()

            # Disparity smoothness loss (edge-aware smoothness)
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)
            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)

            total_loss += loss
            losses[f"loss/{scale}"] = loss.item()
        total_loss = total_loss / self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute evaluation metrics if ground truth depth is available (for logging only)."""
        depth_pred = outputs[("depth", 0, 0)].clamp(min=self.opt.min_depth, max=self.opt.max_depth)
        depth_gt = inputs["depth_gt"]
        # Align predicted depth to median of ground truth depth
        if not self.opt.disable_median_scaling:
            depth_pred *= torch.median(depth_gt) / (torch.median(depth_pred) + 1e-7)
        depth_pred = depth_pred.cpu().numpy()
        depth_gt = depth_gt.cpu().numpy()
        mask = depth_gt > 0
        depth_pred = depth_pred[mask]
        depth_gt = depth_gt[mask]
        # Clamp predictions to min/max depth
        depth_pred[depth_pred < self.opt.min_depth] = self.opt.min_depth
        depth_pred[depth_pred > self.opt.max_depth] = self.opt.max_depth
        # Compute standard depth metrics
        abs_rel = np.mean(np.abs(depth_gt - depth_pred) / depth_gt)
        sq_rel = np.mean(((depth_gt - depth_pred) ** 2) / depth_gt)
        rms = np.sqrt(np.mean((depth_gt - depth_pred) ** 2))
        rms_log = np.sqrt(np.mean((np.log(depth_gt) - np.log(depth_pred)) ** 2))
        a1 = np.mean((np.maximum(depth_gt / depth_pred, depth_pred / depth_gt) < 1.25).astype(float))
        a2 = np.mean((np.maximum(depth_gt / depth_pred, depth_pred / depth_gt) < 1.25 ** 2).astype(float))
        a3 = np.mean((np.maximum(depth_gt / depth_pred, depth_pred / depth_gt) < 1.25 ** 3).astype(float))
        losses["de/abs_rel"] = abs_rel
        losses["de/sq_rel"] = sq_rel
        losses["de/rms"] = rms
        losses["de/log_rms"] = rms_log
        losses["da/a1"] = a1
        losses["da/a2"] = a2
        losses["da/a3"] = a3

    def colormap(self, inputs, normalize=True, torch_transpose=True):
        """Apply the plasma colormap to a depth or disparity array for visualization."""
        # Convert input to numpy array
        if isinstance(inputs, torch.Tensor):
            vis = inputs.detach().cpu().numpy()
        else:
            vis = np.array(inputs, copy=False)
        if normalize:
            ma = float(vis.max()); mi = float(vis.min())
            d = ma - mi if ma != mi else 1e5
            vis = (vis - mi) / (d if d != 0 else 1)
        # Handle different input shapes
        if vis.ndim == 4:
            # (B, C, H, W) -> apply colormap per image
            vis = vis.transpose(0, 2, 3, 1)
            vis = _DEPTH_COLORMAP(vis)
            vis = vis[:, :, :, 0, :3]
            if torch_transpose:
                vis = vis.transpose(0, 3, 1, 2)
        elif vis.ndim == 3:
            # (B, H, W)
            vis = _DEPTH_COLORMAP(vis)
            vis = vis[:, :, :, :3]
            if torch_transpose:
                vis = vis.transpose(0, 3, 1, 2)
        elif vis.ndim == 2:
            # (H, W)
            vis = _DEPTH_COLORMAP(vis)
            vis = vis[..., :3]
            if torch_transpose:
                vis = vis.transpose(2, 0, 1)
        else:
            raise ValueError("Unsupported input shape for colormap")
        return vis
