from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from collections import OrderedDict

class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        # Squeeze the last encoder features of each input to 256 channels
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        # Two convolutional layers to process concatenated features (for multiple inputs)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        # Final convolution: output 6 DOF (axisangle + translation) for each predicted frame
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()
        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        """
        Inputs:
            input_features: list of features from one or more input images.
                            - If num_input_features > 1 (shared encoder case), this should be a list [feat_list_0, feat_list_1, ...].
                            - If num_input_features == 1 (separate encoder case), this should be a list containing a single feature list.
        Outputs:
            axisangle: B x num_frames_to_predict_for x 1 x 3 (rotation vectors)
            translation: B x num_frames_to_predict_for x 1 x 3 (translation vectors)
        """
        # Extract the last-scale features from each input
        last_features = [f[-1] for f in input_features]
        # Squeeze each to 256 channels and apply ReLU
        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        # Concatenate along the channel dimension
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        # Apply the pose conv layers
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)
        # Global average pooling (spatial)
        out = out.mean(3).mean(2)
        # Reshape and scale outputs
        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        # Split into axisangle (rotations) and translation components
        axisangle = out[..., :3]
        translation = out[..., 3:]
        return axisangle, translation
