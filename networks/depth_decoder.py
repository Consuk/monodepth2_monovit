from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from layers import Conv3x3, ConvBlock, upsample


class DepthDecoder(nn.Module):
    """Monodepth2-style depth decoder.

    Expects `input_features` to be a list of 5 feature maps ordered from
    high-resolution (early) to low-resolution (deepest), e.g.:
        [f0, f1, f2, f3, f4]
    where f4 is the deepest.

    `num_ch_enc` must match the channels of these feature maps exactly.
    """

    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super().__init__()

        if not isinstance(num_ch_enc, (list, tuple)) or len(num_ch_enc) != 5:
            raise ValueError(f"DepthDecoder expects num_ch_enc with length 5, got {num_ch_enc}")

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = "nearest"
        self.scales = list(scales)

        self.num_ch_enc = list(num_ch_enc)
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
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        if not isinstance(input_features, (list, tuple)) or len(input_features) != 5:
            raise ValueError(f"DepthDecoder expects input_features list length 5, got {type(input_features)} len={len(input_features) if hasattr(input_features,'__len__') else 'n/a'}")

        outputs = {}

        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = upsample(x)

            if self.use_skips and i > 0:
                skip = input_features[i - 1]
                if skip.shape[2:] != x.shape[2:]:
                    # Nearest is cheaper and avoids align_corners issues
                    skip = F.interpolate(skip, size=x.shape[2:], mode=self.upsample_mode)
                x = torch.cat([x, skip], dim=1)

            x = self.convs[("upconv", i, 1)](x)

            if i in self.scales:
                outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return outputs
