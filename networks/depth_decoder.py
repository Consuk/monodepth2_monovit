from __future__ import absolute_import, division, print_function

"""
Standard Monodepth2-style depth decoder.

This decoder is robust for arbitrary encoders, including transformer-based
backbones like MPViT, because it uses the provided list of channel sizes
returned by the encoder and interpolates skip features when their spatial
sizes differ from the upsampled tensors.  It outputs a dictionary of
disparities at the scales specified when constructing the module.

Borrowed from the original monodepth2 implementation and adapted here
so that users can avoid the more complex high‑resolution decoder
(`DepthDecoderT`) when training with MPViT‑Small.  If you see channel
dimension mismatches when using `DepthDecoderT`, switching to this
decoder should resolve those errors.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from networks.hr_layers import upsample, ConvBlock, Conv3x3  # reuse layers from hr_layers


class DepthDecoder(nn.Module):
    """Simplified depth decoder for Monodepth2.

    Parameters
    ----------
    num_ch_enc : list[int]
        The number of channels for each of the five feature maps returned by
        the encoder.  The list must have length 5 and its order must match
        the order of features returned by the encoder.
    scales : iterable of int, optional
        Scales at which to output disparity maps.  By default it will
        output at scales `[0, 1, 2, 3]`, corresponding to full resolution
        (0) and successive downsampled resolutions.
    num_output_channels : int, optional
        Number of channels in the disparity output.  Defaults to 1.
    use_skips : bool, optional
        Whether to concatenate skip connections from the encoder into the
        decoder.  Skips are strongly recommended for best performance.
    """

    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        # ensure scales is a list so that membership checks work
        self.scales = list(scales)

        self.num_ch_enc = list(num_ch_enc)
        if len(self.num_ch_enc) != 5:
            raise ValueError(
                f"DepthDecoder expects 5 encoder features, got {len(self.num_ch_enc)}"
            )

        # Channel sizes for decoder layers (from smallest to largest scale).
        # These values follow the original monodepth2 implementation.
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        # self.num_ch_dec = np.array([64, 128, 216, 288, 288])

        # Build convolutional blocks for upsampling and feature fusion
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0: first upsampling at this level
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1: optional skip concatenation and refinement
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        # convolutional layers to predict disparity at each scale
        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        # Register all convs as a ModuleList for PyTorch
        self.decoder = nn.ModuleList(list(self.convs.values()))

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        """Decode a list of encoder features into multiscale disparities.

        Parameters
        ----------
        input_features : list[Tensor]
            List of five feature maps from the encoder, ordered from
            lowest‑resolution (deepest) to highest.  Each tensor is of
            shape `(batch_size, channels, height, width)`.

        Returns
        -------
        outputs : dict
            Dictionary mapping keys `("disp", scale)` to predicted
            disparity tensors.  The disparity tensors are passed through a
            sigmoid to ensure values in (0,1).
        """
        if len(input_features) != 5:
            raise ValueError(
                f"DepthDecoder forward expects 5 features, got {len(input_features)}"
            )

        outputs = {}

        # Start from the deepest feature
        x = input_features[-1]
        for i in range(4, -1, -1):
            # upconv_0: process and upsample
            x = self.convs[("upconv", i, 0)](x)
            x = upsample(x)

            # Optionally add skip connection from encoder
            if self.use_skips and i > 0:
                skip = input_features[i - 1]
                if skip.shape[2:] != x.shape[2:]:
                    skip = F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)
                x = torch.cat([x, skip], 1)

            # upconv_1: refine after concatenation
            x = self.convs[("upconv", i, 1)](x)

            # Output disparity at this scale, if requested
            if i in self.scales:
                outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return outputs