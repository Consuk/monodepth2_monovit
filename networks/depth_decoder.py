from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from layers import ConvBlock, Conv3x3, upsample


class DepthDecoder(nn.Module):
    """
    Standard Monodepth2-style depth decoder.

    This version is safe for non-ResNet encoders (e.g., MPViT) because it:
    - uses the provided num_ch_enc list (must match encoder output feature list order)
    - interpolates skip features if their spatial size doesn't match the upsampled tensor
    """
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.scales = list(scales)

        self.num_ch_enc = list(num_ch_enc)
        if len(self.num_ch_enc) != 5:
            raise ValueError(f"DepthDecoder expects 5 encoder features, got {len(self.num_ch_enc)}")

        # Monodepth2 decoder channel sizes
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        if len(input_features) != 5:
            raise ValueError(f"DepthDecoder forward expects 5 features, got {len(input_features)}")

        outputs = {}

        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = upsample(x)

            if self.use_skips and i > 0:
                skip = input_features[i - 1]
                if skip.shape[2:] != x.shape[2:]:
                    skip = F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)
                x = torch.cat([x, skip], 1)

            x = self.convs[("upconv", i, 1)](x)

            if i in self.scales:
                outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return outputs
