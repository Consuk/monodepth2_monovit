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

    IMPORTANT: `input_features` must be a list of 5 feature maps ordered from
    high resolution -> low resolution, i.e.:

        [f0, f1, f2, f3, f4]
        where f4 is the smallest (deepest) feature map.

    For MPViT, the encoder returns exactly this ordering in `forward_features()`.
    """

    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        # Monodepth2 default decoder widths (good quality). If you still OOM,
        # change to e.g. [8, 16, 32, 64, 128].
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.convs = OrderedDict()

        # Decoder: i = 4..0
        for i in range(4, -1, -1):
            # upconv_{i}_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_{i}_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        # Disparity prediction at each requested scale
        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        outputs = {}

        x = input_features[-1]  # smallest feature map
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = upsample(x)

            if self.use_skips and i > 0:
                skip = input_features[i - 1]
                # Spatial alignment (Hamlyn images sometimes have off-by-1 due to resizing/cropping)
                if skip.shape[2:] != x.shape[2:]:
                    # Use nearest to reduce memory vs bilinear
                    skip = F.interpolate(skip, size=x.shape[2:], mode="nearest")
                x = torch.cat([x, skip], 1)

            x = self.convs[("upconv", i, 1)](x)

            if i in self.scales:
                outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return outputs
