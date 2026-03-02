from __future__ import absolute_import, division, print_function

"""
Custom Trainer for Monodepth2 using the MPViT‑Small encoder (MonoViT) and the
standard Monodepth2 depth decoder. This trainer is based on the original
Niantic Monodepth2 trainer but modified to:

* Use the MPViT‑Small transformer encoder for depth estimation.
* Set the encoder channel list to `[64, 128, 216, 288, 288]` to match
  MPViT‑Small output dimensions.
* Instantiate the standard DepthDecoder instead of the high‑resolution
  transformer decoder. This avoids channel mismatches and allows the
  encoder features to be decoded properly.
* Use a ResNet pose encoder and pose decoder for the pose network.
* Compute multiscale reprojection and smoothness losses in the same way
  as the original Monodepth2 implementation, ensuring target images and
  smoothness reference images are at the appropriate scale.

To train with this trainer, instantiate it in `train.py` instead of the
default trainer. For example:

```
from trainer_monovit import Trainer
...
trainer = Trainer(opts)
trainer.train()
```
"""

import os
import time
import json

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from layers import (
    SSIM,
    BackprojectDepth,
    Project3D,
    disp_to_depth,
    get_smooth_loss,
    transformation_from_parameters,
)
from utils import readlines, sec_to_hm_str

import datasets
import networks
import wandb


class Trainer:
    """Trainer object for MonoViT with the Monodepth2 training loop."""

    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # Ensure height and width are multiples of 32 for the decoder
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = (
            2 if self.opt.pose_model_input == "pairs" else self.num_input_frames
        )

        self.models = {}
        self.parameters_to_train = []

        # ------------------------------------------------------------------
        # Build models
        # ------------------------------------------------------------------
        # 1. Encoder: MPViT‑Small for depth estimation
        self.models["encoder"] = networks.mpvit_small()
        # Fix the channel dimensions for MPViT‑Small
        self.models["encoder"].num_ch_enc = [64, 128, 216, 288, 288]
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        # 2. Depth decoder: standard Monodepth2 decoder
        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc,
            self.opt.scales,
        )
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        # 3. Pose network: use a separate ResNet encoder and pose decoder
        self.use_pose_net = not (
            self.opt.use_stereo and self.opt.frame_ids == [0]
        )
        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                # Pose encoder: ResNet
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames,
                )
                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(
                    self.models["pose_encoder"].parameters()
                )
                # Set the channel sizes for ResNet pose encoder
                self.models["pose_encoder"].num_ch_enc = [64, 64, 128, 256, 512]

                # Pose decoder
                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2,
                )
                self.models["pose"].to(self.device)
                self.parameters_to_train += list(self.models["pose"].parameters())
            elif self.opt.pose_model_type == "shared":
                # Use the depth encoder features for pose
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc,
                    self.num_pose_frames,
                )
                self.models["pose"].to(self.device)
                self.parameters_to_train += list(self.models["pose"].parameters())
            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames
                    if self.opt.pose_model_input == "all"
                    else 2
                )
                self.models["pose"].to(self.device)
                self.parameters_to_train += list(self.models["pose"].parameters())

        # Predictive mask network (optional)
        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, (
                "When using predictive_mask, please disable automasking with --disable_automasking"
            )
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc,
                self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1),
            )
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(
                self.models["predictive_mask"].parameters()
            )

        # ------------------------------------------------------------------
        # Optimizer and scheduler
        # ------------------------------------------------------------------
        self.model_optimizer = optim.Adam(
            self.parameters_to_train, self.opt.learning_rate
        )
        self.model_lr_scheduler = None

        # Load pretrained weights if specified
        if self.opt.load_weights_folder is not None:
            self.load_model()

        # ------------------------------------------------------------------
        # Datasets and dataloaders
        # ------------------------------------------------------------------
        # Define dataset mapping. Include 'hamlyn' for HamlynDataset support.
        # Hamlyn dataset is used for endoscopy or Hamlyn-specific depth data.
        datasets_dict = {
            "kitti": datasets.KITTIRAWDataset,
            "kitti_odom": datasets.KITTIOdomDataset,
            "endovis": datasets.SCAREDDataset,
            "hamlyn": datasets.HamlynDataset if hasattr(datasets, "HamlynDataset") else None,
        }
        if self.opt.dataset not in datasets_dict or datasets_dict[self.opt.dataset] is None:
            raise KeyError(
                f"Dataset '{self.opt.dataset}' not found in datasets_dict. "
                "Please ensure it is implemented in the datasets package."
            )
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(
            os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt"
        )
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = ".png" if self.opt.png else ".jpg"

        num_train_samples = len(train_filenames)
        self.num_total_steps = (
            num_train_samples // self.opt.batch_size * self.opt.num_epochs
        )

        train_dataset = self.dataset(
            self.opt.data_path,
            train_filenames,
            self.opt.height,
            self.opt.width,
            self.opt.frame_ids,
            4,
            is_train=True,
            img_ext=img_ext,
        )
        self.train_loader = DataLoader(
            train_dataset,
            self.opt.batch_size,
            True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_dataset = self.dataset(
            self.opt.data_path,
            val_filenames,
            self.opt.height,
            self.opt.width,
            self.opt.frame_ids,
            4,
            is_train=False,
            img_ext=img_ext,
        )
        self.val_loader = DataLoader(
            val_dataset,
            self.opt.batch_size,
            False,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        self.val_iter = iter(self.val_loader)

        # ------------------------------------------------------------------
        # Additional utils
        # ------------------------------------------------------------------
        if not self.opt.no_ssim:
            self.ssim = SSIM().to(self.device)

        # Precompute backproject and project modules per scale
        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)
            self.backproject_depth[scale] = BackprojectDepth(
                self.opt.batch_size, h, w
            ).to(self.device)
            self.project_3d[scale] = Project3D(
                self.opt.batch_size, h, w
            ).to(self.device)

        # Names for depth metrics
        self.depth_metric_names = [
            "de/abs_rel",
            "de/sq_rel",
            "de/rms",
            "de/log_rms",
            "da/a1",
            "da/a2",
            "da/a3",
        ]

        # Log some info
        print("Using dataset split:", self.opt.split)
        print(
            f"There are {len(train_dataset)} training items and {len(val_dataset)} validation items"
        )
        print("Training model named:", self.opt.model_name)
        print("Models and logs will be saved to:", self.log_path)
        print("Using device:", self.device)

        # Save options to disk
        self.save_opts()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def set_train(self):
        for m in self.models.values():
            m.train()

    def set_eval(self):
        for m in self.models.values():
            m.eval()

    def train(self):
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        print(f"Epoch {self.epoch+1}/{self.opt.num_epochs} - Training")
        self.set_train()
        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()
            outputs, losses = self.process_batch(inputs)
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()
            duration = time.time() - before_op_time
            # Logging training progress periodically
            if (
                batch_idx % self.opt.log_frequency == 0
                and self.step < 2000
            ) or (self.step % 2000 == 0):
                self.log_time(batch_idx, duration, losses["loss"])
                # Optionally log depth metrics on validation batch
                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)
                # Log images and losses to wandb
                self.log("train", inputs, outputs, losses)
                self.val()
            self.step += 1
        if self.model_lr_scheduler is not None:
            self.model_lr_scheduler.step()

    def process_batch(self, inputs):
        # Move all inputs to device
        for key, val in inputs.items():
            inputs[key] = val.to(self.device)

        outputs = {}
        # Depth encoder and decoder
        if self.opt.pose_model_type == "shared":
            # Shared encoder: run each frame separately
            all_color_aug = torch.cat(
                [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids], 0
            )
            all_features = self.models["encoder"](all_color_aug)
            all_features = [
                torch.split(f, self.opt.batch_size) for f in all_features
            ]
            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]
            outputs = self.models["depth"](features[0])
        else:
            # Only use frame 0 for depth
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["depth"](features)

        # Predictive mask
        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        # Pose network
        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        # Reconstruct images and compute losses
        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)
        return outputs, losses

    def predict_poses(self, inputs, features):
        outputs = {}
        if self.num_pose_frames == 2:
            # Compute pose for each source frame via separate forward passes
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {
                    f_i: inputs["color_aug", f_i, 0]
                    for f_i in self.opt.frame_ids
                }
            for f_i in self.opt.frame_ids[1:]:
                if f_i == "s":
                    continue
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]
                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                elif self.opt.pose_model_type == "posecnn":
                    pose_inputs = torch.cat(pose_inputs, 1)
                axisangle, translation = self.models["pose"](pose_inputs)
                outputs[("axisangle", 0, f_i)] = axisangle
                outputs[("translation", 0, f_i)] = translation
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=(f_i < 0)
                )
        else:
            # Multi-frame input to the pose network
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [
                        inputs["color_aug", i, 0]
                        for i in self.opt.frame_ids
                        if i != "s"
                    ],
                    1,
                )
                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]
            elif self.opt.pose_model_type == "shared":
                pose_inputs = [
                    features[i] for i in self.opt.frame_ids if i != "s"
                ]
            axisangle, translation = self.models["pose"](pose_inputs)
            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i == "s":
                    continue
                outputs[("axisangle", 0, f_i)] = axisangle
                outputs[("translation", 0, f_i)] = translation
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, i], translation[:, i], invert=(f_i < 0)
                )
        return outputs

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                # Upsample disp to full resolution
                disp = F.interpolate(
                    disp,
                    [self.opt.height, self.opt.width],
                    mode="bilinear",
                    align_corners=False,
                )
                source_scale = 0
            # Convert disp to depth
            _, depth = disp_to_depth(
                disp, self.opt.min_depth, self.opt.max_depth
            )
            outputs[("depth", 0, scale)] = depth
            for f_i in self.opt.frame_ids[1:]:
                if f_i == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, f_i)]
                # If using posecnn, adjust T based on inverse depth
                if self.opt.pose_model_type == "posecnn":
                    axisangle = outputs[("axisangle", 0, f_i)]
                    translation = outputs[("translation", 0, f_i)]
                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)
                    T = transformation_from_parameters(
                        axisangle[:, 0],
                        translation[:, 0] * mean_inv_depth[:, 0],
                        invert=(f_i < 0),
                    )
                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)]
                )
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T
                )
                outputs[("sample", f_i, scale)] = pix_coords
                outputs[("color", f_i, scale)] = F.grid_sample(
                    inputs[("color", f_i, source_scale)],
                    outputs[("sample", f_i, scale)],
                    padding_mode="border",
                    align_corners=True,
                )
                if not self.opt.disable_automasking:
                    outputs[("color_identity", f_i, scale)] = inputs[("color", f_i, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        if self.opt.no_ssim:
            return l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            return 0.85 * ssim_loss + 0.15 * l1_loss

    def compute_losses(self, inputs, outputs):
        """Compute the photometric reprojection and smoothness losses."""
        losses = {}
        total_loss = 0.0
        for scale in self.opt.scales:
            reprojection_losses = []
            # Determine the scale of the target image
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0
            # Target image at source_scale
            target = inputs[("color", 0, source_scale)]
            # Accumulate reprojection losses from each source frame
            for frame_id in self.opt.frame_ids[1:]:
                if frame_id == "s":
                    continue
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)
            # Automasking: compare against identity reprojection
            if not self.opt.disable_automasking:
                identity_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    if frame_id == "s":
                        continue
                    pred_id = inputs[("color", frame_id, source_scale)]
                    identity_losses.append(self.compute_reprojection_loss(pred_id, target))
                identity_losses = torch.cat(identity_losses, 1)
                # Optionally average reprojection losses
                if self.opt.avg_reprojection:
                    reprojection_losses = reprojection_losses.mean(1, keepdim=True)
                    identity_losses = identity_losses.mean(1, keepdim=True)
                # Add random noise to break ties
                identity_losses = identity_losses + (
                    torch.randn_like(identity_losses) * 1e-5
                )
                combined = torch.cat([identity_losses, reprojection_losses], dim=1)
                if combined.shape[1] == 1:
                    to_optimise = combined
                else:
                    to_optimise, _ = torch.min(combined, dim=1, keepdim=True)
            else:
                # Without automasking, just take the minimum reprojection error
                if self.opt.avg_reprojection:
                    reprojection_losses = reprojection_losses.mean(1, keepdim=True)
                to_optimise, _ = torch.min(reprojection_losses, dim=1, keepdim=True)
            loss = to_optimise.mean()
            # Smoothness loss
            disp = outputs[("disp", scale)]
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            # Ensure the reference image used for smoothness has the same spatial size
            if target.shape[-2:] != disp.shape[-2:]:
                target_s = F.interpolate(
                    target,
                    size=disp.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                target_s = target
            smooth_loss = get_smooth_loss(norm_disp, target_s)
            loss = loss + (self.opt.disparity_smoothness * smooth_loss) / (2 ** scale)
            total_loss = total_loss + loss
            losses[f"loss/{scale}"] = loss.detach()
        total_loss = total_loss / self.num_scales
        losses["loss"] = total_loss
        return losses

    def val(self):
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
            del inputs, outputs, losses
        self.set_train()

    # ------------------------------------------------------------------
    # Depth metrics (optional)
    # ------------------------------------------------------------------
    def compute_depth_losses(self, inputs, outputs, losses):
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(
            F.interpolate(
                depth_pred,
                [375, 1242],
                mode="bilinear",
                align_corners=False,
            ),
            1e-3,
            80,
        )
        depth_pred = depth_pred.detach()
        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask
        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred = depth_pred * (torch.median(depth_gt) / torch.median(depth_pred))
        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)
        depth_errors = utils.compute_depth_errors(depth_gt, depth_pred)
        for i, name in enumerate(self.depth_metric_names):
            losses[name] = float(depth_errors[i].cpu())

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def log_time(self, batch_idx, duration, loss):
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / max(1, self.step) - 1.0
        ) * time_sofar
        print(
            f"epoch {self.epoch+1:>3} | batch {batch_idx:>6} | examples/s: {samples_per_sec:5.1f}"
            f" | loss: {loss.item():.5f} | time elapsed: {sec_to_hm_str(time_sofar)}"
            f" | time left: {sec_to_hm_str(training_time_left)}"
        )

    def log(self, mode, inputs, outputs, losses):
        # Prepare data for wandb logging
        log_data = {f"{mode}{k}": float(v.detach().cpu()) if torch.is_tensor(v) else float(v)
                    for k, v in losses.items()}
        # Pick an index for visualization
        s = 0  # highest resolution
        B = inputs[("color", 0, s)].shape[0]
        viz_idx = (self.step // max(1, self.opt.log_frequency)) % B
        img0 = inputs[("color", 0, s)][viz_idx].detach().cpu()
        log_data[f"{mode}color/0_{s}"] = wandb.Image(img0)
        for frame_id in self.opt.frame_ids:
            if frame_id == 0:
                continue
            img_src = inputs[("color", frame_id, s)][viz_idx].detach().cpu()
            log_data[f"{mode}color/{frame_id}_{s}"] = wandb.Image(img_src)
            if ("color", frame_id, s) in outputs:
                img_pred = outputs[("color", frame_id, s)][viz_idx].detach().cpu()
                log_data[f"{mode}color_pred/{frame_id}_{s}"] = wandb.Image(img_pred)
        disp = outputs[("disp", s)][viz_idx, 0].detach().cpu().numpy()
        disp_vis = self.colormap(disp).transpose(1, 2, 0)
        log_data[f"{mode}disp/{s}"] = wandb.Image(disp_vis)
        wandb.log(log_data, step=self.step)

    # ------------------------------------------------------------------
    # Saving and loading models
    # ------------------------------------------------------------------
    def save_opts(self):
        models_dir = os.path.join(self.log_path, "models")
        os.makedirs(models_dir, exist_ok=True)
        to_save = self.opt.__dict__.copy()
        with open(os.path.join(models_dir, "opt.json"), "w") as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        print("Saving models to", self.log_path)
        save_folder = os.path.join(self.log_path, "models", f"weights_{self.epoch}")
        os.makedirs(save_folder, exist_ok=True)
        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, f"{model_name}.pth")
            to_save = model.state_dict()
            if model_name in ["encoder", "pose_encoder"]:
                to_save = {**to_save, "height": self.opt.height, "width": self.opt.width}
            torch.save(to_save, save_path)
        torch.save(self.model_optimizer.state_dict(), os.path.join(save_folder, "adam.pth"))

    def load_model(self):
        load_folder = self.opt.load_weights_folder
        print(f"Loading model from folder {load_folder}")
        for n in self.opt.models_to_load:
            if n not in self.models:
                print(f"  [load_model] Skipping {n}: not in current model dict")
                continue
            print(f"  Loading {n} weights...")
            path = os.path.join(load_folder, f"{n}.pth")
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict
            }
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)
        # Load optimizer state if available
        opt_path = os.path.join(load_folder, "adam.pth")
        if os.path.isfile(opt_path):
            print("  Loading Adam state")
            self.model_optimizer.load_state_dict(torch.load(opt_path))
        else:
            print("  Adam state not found; optimizer will be randomly initialized")

    def colormap(self, inputs, normalize=True, torch_transpose=True):
        import matplotlib.pyplot as plt
        _DEPTH_COLORMAP = plt.get_cmap("plasma", 256)
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.detach().cpu().numpy()
        vis = inputs
        if normalize:
            ma = float(vis.max())
            mi = float(vis.min())
            d = ma - mi if ma != mi else 1e5
            vis = (vis - mi) / d
        if vis.ndim == 4:
            vis = vis.transpose([0, 2, 3, 1])
            vis = _DEPTH_COLORMAP(vis)
            vis = vis[:, :, :, 0, :3]
            if torch_transpose:
                vis = vis.transpose(0, 3, 1, 2)
        elif vis.ndim == 3:
            vis = _DEPTH_COLORMAP(vis)
            vis = vis[:, :, :, :3]
            if torch_transpose:
                vis = vis.transpose(0, 3, 1, 2)
        elif vis.ndim == 2:
            vis = _DEPTH_COLORMAP(vis)
            vis = vis[..., :3]
            if torch_transpose:
                vis = vis.transpose(2, 0, 1)
        return vis