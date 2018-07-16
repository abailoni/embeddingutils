from inferno.extensions.containers.graph import Graph, Identity

from inferno.extensions.layers.convolutional import ConvELU3D, Conv3D, BNReLUConv3D
from inferno.extensions.layers.sampling import AnisotropicPool, AnisotropicUpsample
from inferno.extensions.layers.reshape import Concatenate, Sum

import torch
import torch.nn as nn

import numpy as np

# TODO: support residual U-Net, Hourglass-Net, 2D Versions, stacked Versions


class EncoderDecoderSkeleton(Graph):
    def __init__(self, depth):
        super(EncoderDecoderSkeleton, self).__init__()
        self.depth = depth

    def setup_graph(self):
        depth = self.depth
        self.add_input_node('input')
        for i in range(depth):
            self.add_node(f'encoder_{i}', self.construct_encoder_module(i),
                          previous='input' if i == 0 else f'down_{i-1}_{i}')
            self.add_node(f'skip_{i}', self.construct_skip_module(i), previous=f'encoder_{i}')
            self.add_node(f'down_{i}_{i+1}', self.construct_downsampling_module(i), previous=f'encoder_{i}')

        self.add_node('base', self.construct_base_module(), previous=f'down_{depth-1}_{depth}')

        for i in reversed(range(depth)):
            self.add_node(f'up_{i+1}_{i}', self.construct_upsampling_module(i),
                          previous='base' if i == depth - 1 else f'decoder_{i+1}')
            self.add_node(f'merge_{i}', self.construct_merge_module(i), previous=[f'skip_{i}', f'up_{i+1}_{i}'])
            self.add_node(f'decoder_{i}', self.construct_decoder_module(i), previous=f'merge_{i}')

        self.add_node('final', self.construct_output_module(), previous=f'decoder_0')
        self.add_output_node('output', previous='final')

    def construct_encoder_module(self, depth):
        return Identity()

    def construct_decoder_module(self, depth):
        return self.construct_encoder_module(depth)

    def construct_downsampling_module(self, depth):
        return Identity()

    def construct_upsampling_module(self, depth):
        return Identity()

    def construct_skip_module(self, depth):
        return Identity()

    def construct_merge_module(self, depth):
        return Concatenate()

    def construct_base_module(self):
        return Identity()

    def construct_output_module(self):
        return Identity()


class UNetSkeleton(EncoderDecoderSkeleton):
    def __init__(self, depth, in_channels, out_channels, fmaps=None, **kwargs):
        super(UNetSkeleton, self).__init__(depth)
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(fmaps, (list, tuple)):
            self.fmaps = fmaps
        else:
            assert isinstance(fmaps, int)
            if 'fmap_increase' in kwargs:
                self.fmap_increase = kwargs['fmap_increase']
                self.fmaps = [fmaps + i * self.fmap_increase for i in range(self.depth+1)]
            elif 'fmap_factor' in kwargs:
                self.fmap_factor = kwargs['fmap_factor']
                self.fmaps = [fmaps * i**self.fmap_factor for i in range(self.depth + 1)]
            else:
                self.fmaps = [fmaps, ] * (self.depth + 1)
        assert len(self.fmaps) == self.depth + 1

        self.merged_fmaps = [2*n for n in self.fmaps]

    def construct_conv(self, f_in, f_out):
        pass

    def construct_merge_module(self, depth):
        return Concatenate()

    def construct_encoder_module(self, depth):
        f_in = self.in_channels if depth == 0 else self.fmaps[depth - 1]
        f_out = self.fmaps[depth]
        return nn.Sequential(
            self.construct_conv(f_in, f_out),
            self.construct_conv(f_out, f_out)
        )

    def construct_decoder_module(self, depth):
        f_in = self.merged_fmaps[depth]
        f_intermediate = self.fmaps[depth]
        f_out = self.out_channels if depth == 0 else self.fmaps[depth - 1]
        return nn.Sequential(
            self.construct_conv(f_in, f_intermediate),
            self.construct_conv(f_intermediate, f_out)
        )

    def construct_base_module(self):
        f_in = self.fmaps[self.depth-1]
        f_intermediate = self.fmaps[self.depth]
        f_out = self.fmaps[self.depth-1]
        return nn.Sequential(
            self.construct_conv(f_in, f_intermediate),
            self.construct_conv(f_intermediate, f_out)
        )


CONV_TYPES = {'vanilla': ConvELU3D,
              'conv_bn': BNReLUConv3D}


class UNet3D(UNetSkeleton):
    def __init__(self,
                 scale_factor=2,
                 conv_type='vanilla',
                 final_activation=None,
                 *super_args, **super_kwargs):

        super(UNet3D, self).__init__(*super_args, **super_kwargs)

        self.final_activation = final_activation

        # parse conv_type
        if isinstance(conv_type, str):
            assert conv_type in CONV_TYPES
            self.conv_type = CONV_TYPES[conv_type]
        else:
            assert isinstance(conv_type, type)
            self.conv_type = conv_type

        # parse scale factor
        if isinstance(scale_factor, int):
            scale_factor = [scale_factor, ] * super_kwargs['depth']
        scale_factors = scale_factor
        normalized_factors = []
        for scale_factor in scale_factors:
            assert isinstance(scale_factor, (int, list, tuple))
            if isinstance(scale_factor, int):
                scale_factor = 3 * [scale_factor, ]
            assert len(scale_factor) == 3
            normalized_factors.append(scale_factor)
        self.scale_factors = normalized_factors

        # compute input size divisibiliy constraints
        divisibility_constraint = np.ones(3)
        for scale_factor in self.scale_factors:
            divisibility_constraint *= np.array(scale_factor)
        self.divisibility_constraint = list(divisibility_constraint.astype(int))

        self.setup_graph()  # TODO: this is sooo ugly. do it when forward() is called for the first time?

    def construct_conv(self, f_in, f_out):
        return self.conv_type(f_in, f_out, kernel_size=3)

    def construct_downsampling_module(self, depth):
        scale_factor = self.scale_factors[depth]
        sampler = nn.MaxPool3d(kernel_size=scale_factor,
                               stride=scale_factor,
                               padding=0)
        return sampler

    def construct_upsampling_module(self, depth):
        scale_factor = self.scale_factors[depth]
        if scale_factor[0] == 1:
            assert scale_factor[1] == scale_factor[2]
            sampler = AnisotropicUpsample(scale_factor=scale_factor[1])
        else:
            sampler = nn.Upsample(scale_factor=scale_factor[0])
        return sampler

    def construct_output_module(self):
        if self.final_activation is not None:
            return self.final_activation
        else:
            return Identity()

    def forward(self, input_):
        assert all(input_.shape[-i] % self.divisibility_constraint[-i] == 0 for i in range(1, 4)), \
            f'Volume dimensions {input_.shape[-3:]} are not divisible by {self.divisibility_constraint}'
        return super(UNet3D, self).forward(input_)


if __name__ == '__main__':
    model = UNet3D(depth=3,
                   in_channels=1,
                   out_channels=2,
                   fmaps=5,
                   fmap_increase=1,
                   scale_factor=[[1, 3, 3], 2, 2],
                   final_activation=nn.Sigmoid())

    print(model)
    model = model.cuda()
    inp = torch.ones(model.divisibility_constraint)[None, None].cuda()
    out = model(inp)
    print(inp.shape)
    print(out.shape)

