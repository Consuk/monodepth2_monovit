from __future__ import absolute_import, division, print_function

import os
import time
import json
import subprocess
from collections import OrderedDict

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import datasets
import networks
import wandb

from layers import SSIM, BackprojectDepth, Project3D, disp_to_depth, get_smooth_loss
from networks.hr_layers import transformation_from_parameters



from layers import SSIM, BackprojectDepth, Project3D, disp_to_depth, get_smooth_loss
from utils import readlines, normalize_image, sec_to_hm_str


# -------------------------
# AMP compatibility (torch 1.x / 2.x)
# -------------------------
try:
    from torch.cuda.amp import autocast, GradScaler  # torch<=1.13
except Exception:  # pragma: no cover
    from torch.amp import autocast, GradScaler  # torch>=2.0


def infer_num_ch_enc(encoder, in_chans, height, width, device):
    """
    Forward a dummy tensor through the encoder to infer the feature-channel sizes.
    This is robust for custom backbones (e.g., MPViT) where num_ch_enc isn't fixed.
    """
    encoder.eval()
    with torch.no_grad():
        x = torch.zeros(1, in_chans, height, width, device=device)
        feats = encoder(x)
        if not isinstance(feats, (list, tuple)):
            raise RuntimeError(
                "Encoder forward must return a list/tuple of feature maps. "
                f"Got type={type(feats)}")
        chs = [int(f.shape[1]) for f in feats]
    return chs


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # Ensure input dimensions are multiples of 32 (required by decoder pyramid)
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        self.models = {}
        self.parameters_to_train = []
        self.scaler = GradScaler(enabled=(self.device.type == "cuda"))

        # -------------------------
        # MODELS
        # -------------------------
        # Encoder: MPViT small (monovit)
        # Use mpvit_small to build the encoder instead of mpvit_tiny. This
        # function is imported from mpvit.py and returns a MPViT-small model.
        self.models["encoder"] = networks.mpvit_small()
        self.models["encoder"].to(self.device)

        # Infer encoder channels AFTER moving to device to avoid CPU/GPU dtype mismatch
        # Infer the channel sizes for the encoder by passing a dummy 3‑channel
        # tensor through the model. We then adjust the second entry to match
        # the first if the encoder does not increase the channel dimension
        # between the stem and the first stage (MPViT-Small typically yields
        # [64, 128, ...], but in our depth decoder we need the second element
        # to equal the first when it remains unchanged). This avoids mismatches
        # in the high-resolution decoder when the feature dimensions do not
        # double after the stem.
        chs = infer_num_ch_enc(
            self.models["encoder"], in_chans=3,
            height=self.opt.height, width=self.opt.width,
            device=self.device
        )
        
        self.models["encoder"].num_ch_enc = chs

        # Depth decoder
        self.models["depth"] = networks.DepthDecoder(
            num_ch_enc=self.models["encoder"].num_ch_enc,
            scales=self.opt.scales,
            num_output_channels=1,
            use_skips=True
        ).to(self.device)
        self.models["depth"].to(self.device)

        # Pose network
        if self.opt.pose_model_type == "separate_resnet":
            # MPViT pose encoder over concatenated (target, source) -> 6 channels
            # Use mpvit_small for pose encoder as well to maintain the same
            # architecture family.
            # Instantiate the pose encoder with 6 input channels to handle
            # concatenated (target, source) frames. Without specifying
            # in_chans=6, the encoder defaults to 3 channels and fails when
            # receiving 6‑channel input.
            self.models["pose_encoder"] = networks.mpvit_small(in_chans=6)
            self.models["pose_encoder"].to(self.device)
            # Infer the channel sizes for the pose encoder. Use in_chans=6 to
            # match the input and preserve the dynamic channel sizes of MPViT.
            chs_pose = infer_num_ch_enc(
                self.models["pose_encoder"], in_chans=6,
                height=self.opt.height, width=self.opt.width,
                device=self.device
            )
            if len(chs_pose) >= 2 and chs_pose[1] != chs_pose[0]:
                chs_pose = chs_pose.copy()
                chs_pose[1] = chs_pose[0]
            if len(chs_pose) >= 5 and chs_pose[4] != chs_pose[3]:
                chs_pose[4] = chs_pose[3]
            self.models["pose_encoder"].num_ch_enc = chs_pose
            self.models["pose"] = networks.PoseDecoder(
                self.models["pose_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2
            )
            self.models["pose"].to(self.device)

        elif self.opt.pose_model_type == "shared":
            self.models["pose"] = networks.PoseDecoder(
                self.models["encoder"].num_ch_enc,
                num_input_features=self.num_pose_frames - 1,
                num_frames_to_predict_for=2
            )
            self.models["pose"].to(self.device)

        elif self.opt.pose_model_type == "posecnn":
            self.models["pose"] = networks.PoseCNN(self.num_pose_frames)
            self.models["pose"].to(self.device)
        else:
            raise ValueError(f"Unknown pose_model_type: {self.opt.pose_model_type}")

        # Predictive mask (optional)
        if self.opt.predictive_mask:
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc,
                self.opt.scales,
                num_output_channels=(self.num_input_frames - 1)
            )
            self.models["predictive_mask"].to(self.device)

        # Trainable parameters
        for k, m in self.models.items():
            self.parameters_to_train += list(m.parameters())

        # Optimizer & scheduler
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1
        )

        # SSIM
        self.ssim = SSIM()
        self.ssim.to(self.device)

        # Backproject / project layers
        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)
            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w).to(self.device)
            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w).to(self.device)

        # Load weights if provided
        if self.opt.load_weights_folder is not None:
            self.load_model()

        # -------------------------
        # DATA
        # -------------------------
        splits_dir = os.path.join(os.path.dirname(__file__), "splits", self.opt.split)
        train_fpath = os.path.join(splits_dir, "train_files.txt")
        val_fpath = os.path.join(splits_dir, "val_files.txt")

        if not os.path.exists(val_fpath):
            # Hamlyn split in your repo may not include val_files; fall back to test_files
            test_fpath = os.path.join(splits_dir, "test_files.txt")
            if os.path.exists(test_fpath):
                print(f"[split] WARNING: {val_fpath} not found. Using {test_fpath} as validation.")
                val_fpath = test_fpath
            else:
                raise FileNotFoundError(f"Missing val_files.txt and test_files.txt in {splits_dir}")

        train_filenames = readlines(train_fpath)
        val_filenames = readlines(val_fpath)

        self.num_total_steps = len(train_filenames) // self.opt.batch_size * self.opt.num_epochs

        dataset_dict = {
            "endovis": datasets.SCAREDDataset,
            "hamlyn": datasets.HamlynDataset,
        }
        self.dataset = dataset_dict[self.opt.dataset]

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4,
            is_train=True,
            img_ext=".png" if self.opt.png else ".jpg",
            strict_neighbors=getattr(self.opt, "hamlyn_strict_neighbors", False),
            neighbor_search_max=getattr(self.opt, "neighbor_search_max", 10)
        )
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True
        )

        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4,
            is_train=False,
            img_ext=".png" if self.opt.png else ".jpg",
            strict_neighbors=getattr(self.opt, "hamlyn_strict_neighbors", False),
            neighbor_search_max=getattr(self.opt, "neighbor_search_max", 10)
        )
        self.val_loader = DataLoader(
            val_dataset, self.opt.eval_batch_size, False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True
        )

        print("Training model named:", self.opt.model_name)
        print("Models and logs will be saved to:", self.opt.log_dir)
        print("Using device:", self.device)
        print("Using split:", self.opt.split)
        print("Training samples:", len(train_dataset), "Validation samples:", len(val_dataset))

        self.epoch = 0
        self.step = 0
        self.start_time = time.time()

    # -------------------------
    # TRAIN LOOP
    # -------------------------
    def train(self):
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()

            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

            if getattr(self.opt, "eval_each_epoch", False):
                self.try_eval_each_epoch()

            self.model_lr_scheduler.step()

    def run_epoch(self):
        for m in self.models.values():
            m.train()

        print(f"Epoch {self.epoch + 1}/{self.opt.num_epochs} - Training")

        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            loss = losses["loss"]

            self.model_optimizer.zero_grad(set_to_none=True)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.model_optimizer)
            self.scaler.update()

            duration = time.time() - before_op_time

            # Logging
            if self.step % self.opt.log_frequency == 0:
                self.log_time(batch_idx, duration, loss)
                self.log_wandb(losses)

            self.step += 1

    # -------------------------
    # BATCH
    # -------------------------
    def process_batch(self, inputs):
        for k in list(inputs.keys()):
            inputs[k] = inputs[k].to(self.device)

        with autocast(enabled=(self.device.type == "cuda")):
            features = self.models["encoder"](inputs[("color_aug", 0, 0)])
            # IMPORTANT: do NOT reverse features; DepthDecoder expects [low->high] order.
            outputs = self.models["depth"](features)

            if self.opt.predictive_mask:
                outputs["predictive_mask"] = self.models["predictive_mask"](features)

            if self.opt.pose_model_type == "shared":
                pose_feats = [features]
            else:
                pose_feats = None

            outputs.update(self.predict_poses(inputs, features, pose_feats))

            self.generate_images_pred(inputs, outputs)
            losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    # -------------------------
    # POSE
    # -------------------------
    def predict_poses(self, inputs, features, pose_feats):
        outputs = {}

        if self.num_pose_frames == 2:
            # predict poses for each source frame individually
            for f_i in self.opt.frame_ids[1:]:
                if f_i == "s":
                    continue

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = torch.cat([inputs[("color_aug", 0, 0)], inputs[("color_aug", f_i, 0)]], 1)
                    # The pose encoder returns a list of feature maps.  The pose decoder
                    # expects a list of feature lists (one per input) rather than a
                    # single tensor.  Passing the deepest feature alone ([-1]) will
                    # index into the tensor and drop the batch dimension, leading to
                    # shape mismatches.  Instead, wrap the entire list in another
                    # list so that the decoder can index correctly.
                    pose_feats_i = self.models["pose_encoder"](pose_inputs)
                    axisangle, translation = self.models["pose"]([pose_feats_i])
                elif self.opt.pose_model_type == "shared":
                    # pose_feats already computed
                    axisangle, translation = self.models["pose"](pose_feats, [f_i])
                else:  # posecnn
                    pose_inputs = torch.cat([inputs[("color_aug", f_i, 0)], inputs[("color_aug", 0, 0)]], 1)
                    axisangle, translation = self.models["pose"](pose_inputs)

                outputs[("axisangle", 0, f_i)] = axisangle
                outputs[("translation", 0, f_i)] = translation

                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=(f_i < 0)
                )

        else:
            raise NotImplementedError("pose_model_input='all' not supported in this patched trainer")

        return outputs

    # -------------------------
    # IMAGE SYNTHESIS
    # -------------------------
    def generate_images_pred(self, inputs, outputs):
        """
        Generate the warped (reprojected) images for photometric loss.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if not self.opt.v1_multiscale:
                disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth

            for f_i in self.opt.frame_ids[1:]:
                if f_i == "s":
                    continue

                T = outputs[("cam_T_cam", 0, f_i)]
                cam_points = self.backproject_depth[scale](
                    depth, inputs[("inv_K", 0)]
                )
                pix_coords = self.project_3d[scale](
                    cam_points, inputs[("K", 0)], T
                )
                outputs[("sample", f_i, scale)] = pix_coords
                outputs[("color", f_i, scale)] = F.grid_sample(
                    inputs[("color", f_i, 0)],
                    pix_coords,
                    padding_mode="border",
                    align_corners=True
                )

                if not self.opt.disable_automasking:
                    outputs[("color_identity", f_i, scale)] = inputs[("color", f_i, 0)]

    # -------------------------
    # LOSSES
    # -------------------------
    def compute_reprojection_loss(self, pred, target):
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            target = inputs[("color", 0, 0)]
            pred = outputs[("color", self.opt.frame_ids[1], scale)]
            # NOTE: above line expects frame_ids[1] exists (-1 by default). We'll compute for all frames below.

            # reprojection for each source frame
            for frame_id in self.opt.frame_ids[1:]:
                if frame_id == "s":
                    continue
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    if frame_id == "s":
                        continue
                    pred = inputs[("color", frame_id, 0)]
                    identity_reprojection_losses.append(self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_losses = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    identity_reprojection_losses = identity_reprojection_losses

                if self.opt.avg_reprojection:
                    reprojection_losses = reprojection_losses.mean(1, keepdim=True)

                combined = torch.cat([identity_reprojection_losses, reprojection_losses], dim=1)

                if combined.shape[1] == 1:
                    to_optimise = combined
                else:
                    to_optimise, _ = torch.min(combined, dim=1)
                    to_optimise = to_optimise.unsqueeze(1)
            else:
                if self.opt.avg_reprojection:
                    reprojection_losses = reprojection_losses.mean(1, keepdim=True)
                to_optimise, _ = torch.min(reprojection_losses, dim=1)
                to_optimise = to_optimise.unsqueeze(1)

            loss += to_optimise.mean()

            # smoothness
            disp = outputs[("disp", scale)]
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, target)
            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)

            total_loss += loss
            losses[f"loss/{scale}"] = loss.detach()

        total_loss /= self.num_scales
        losses["loss"] = total_loss

        return losses

    # -------------------------
    # LOGGING
    # -------------------------
    def log_time(self, batch_idx, duration, loss):
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps / max(1, self.step) - 1.0) * time_sofar

        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f} | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(
            self.epoch + 1, batch_idx, samples_per_sec, loss.item(),
            sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)
        ))

    def log_wandb(self, losses):
        log_dict = {}
        for k, v in losses.items():
            if torch.is_tensor(v):
                log_dict[k] = float(v.detach().cpu().item())
            else:
                log_dict[k] = float(v)
        wandb.log(log_dict, step=self.step)

    # -------------------------
    # SAVE/LOAD
    # -------------------------
    def save_model(self):
        save_folder = os.path.join(self.log_path, "models", f"weights_{self.epoch}")
        os.makedirs(save_folder, exist_ok=True)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, f"{model_name}.pth")
            to_save = model.state_dict()
            if model_name == "encoder" or model_name == "pose_encoder":
                # store input size for evaluation
                to_save = {**to_save, "height": self.opt.height, "width": self.opt.width}
            torch.save(to_save, save_path)

        # save opts
        with open(os.path.join(save_folder, "opt.json"), "w") as f:
            json.dump(vars(self.opt), f, indent=2)

    def load_model(self):
        load_folder = self.opt.load_weights_folder
        print(f"Loading model from folder {load_folder}")

        for n in self.opt.models_to_load:
            if n not in self.models:
                print(f"  [load_model] Skipping {n}: not in current model dict")
                continue

            path = os.path.join(load_folder, f"{n}.pth")
            if not os.path.exists(path):
                print(f"  [load_model] Missing: {path}")
                continue

            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path, map_location=self.device)

            # remove metadata keys
            for meta_k in ["height", "width"]:
                if meta_k in pretrained_dict:
                    pretrained_dict.pop(meta_k)

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

    # -------------------------
    # OPTIONAL: EVAL EACH EPOCH
    # -------------------------
    def try_eval_each_epoch(self):
        """
        Runs evaluate_hr_depth.py as a subprocess at the end of each epoch.
        If it fails (e.g., missing GT), training continues.
        """
        weights_folder = os.path.join(self.log_path, "models", f"weights_{self.epoch}")
        if not os.path.exists(weights_folder):
            return

        cmd = [
            "python", "evaluate_hr_depth.py",
            "--data_path", self.opt.data_path,
            "--load_weights_folder", weights_folder,
            "--eval_split", self.opt.eval_split,
            "--eval_mono",
        ]
        # keep strict neighbors for hamlyn eval, if used in training
        if getattr(self.opt, "hamlyn_strict_neighbors", False):
            cmd += ["--hamlyn_strict_neighbors"]

        print("[eval_each_epoch] Running:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            print("[eval_each_epoch] WARNING: evaluation failed:", repr(e))
