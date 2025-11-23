from __future__ import annotations
import logging
import tensorflow.compat.v1 as tf

from .base import ModelBase
from deepfacelab.core.leras.layers.conv2d import Conv2D

logger = logging.getLogger(__name__)


class CodeDiscriminator(ModelBase):
    def on_build(self, in_ch, code_res, ch=256, conv_kernel_initializer=None):
        n_downscales = 1 + code_res // 8

        self.convs = []
        prev_ch = in_ch
        for i in range(n_downscales):
            cur_ch = ch * min((2**i), 8)
            self.convs.append(
                Conv2D(
                    prev_ch,
                    cur_ch,
                    kernel_size=4 if i == 0 else 3,
                    strides=2,
                    padding="SAME",
                    kernel_initializer=conv_kernel_initializer,
                )
            )
            prev_ch = cur_ch

        self.out_conv = Conv2D(
            prev_ch,
            1,
            kernel_size=1,
            padding="VALID",
            kernel_initializer=conv_kernel_initializer,
        )

    def forward(self, x):
        for conv in self.convs:
            x = tf.nn.leaky_relu(conv(x), 0.1)
        return self.out_conv(x)
