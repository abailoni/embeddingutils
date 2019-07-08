from inferno.extensions.containers.graph import Graph, Identity

from inferno.extensions.layers.convolutional import ConvELU3D, Conv3D, BNReLUConv3D
from inferno.extensions.layers.convolutional import ConvELU2D, Conv2D, BNReLUConv2D
from inferno.extensions.layers.sampling import AnisotropicPool, AnisotropicUpsample
from inferno.extensions.layers.reshape import Concatenate, Sum

from embeddingutils.models.submodules import SuperhumanSNEMIBlock, ConvGRU, ShakeShakeMerge, Upsample
from embeddingutils.models.submodules import ResBlockAdvanced

import torch
import torch.nn as nn

import numpy as np

# TODO: support residual U-Net, Hourglass-Net, 2D Versions, stacked Versions


class EncoderDecoderSkeleton(nn.Module):

    def __init__(self, depth):
        super(EncoderDecoderSkeleton, self).__init__()
        self.depth = depth
        # construct all the layers
        self.encoder_modules = nn.ModuleList(
            [self.construct_encoder_module(i) for i in range(depth)])
        self.skip_modules = nn.ModuleList(
            [self.construct_skip_module(i) for i in range(depth)])
        self.downsampling_modules = nn.ModuleList(
            [self.construct_downsampling_module(i) for i in range(depth)])
        self.upsampling_modules = nn.ModuleList(
            [self.construct_upsampling_module(i) for i in range(depth)])
        self.decoder_modules = nn.ModuleList(
            [self.construct_decoder_module(i) for i in range(depth)])
        self.merge_modules = nn.ModuleList(
            [self.construct_merge_module(i) for i in range(depth)])
        self.base_module = self.construct_base_module()
        self.final_module = self.construct_output_module()

    def forward(self, input):
        encoded_states = []
        current = input
        for encode, downsample in zip(self.encoder_modules, self.downsampling_modules):
            current = encode(current)
            encoded_states.append(current)
            current = downsample(current)
        current = self.base_module(current)
        for encoded_state, upsample, skip, merge, decode in reversed(list(zip(
                encoded_states, self.upsampling_modules, self.skip_modules, self.merge_modules, self.decoder_modules))):
            current = upsample(current)
            encoded_state = skip(encoded_state)
            current = merge(current, encoded_state)
            current = decode(current)
        current = self.final_module(current)
        return current

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


class MergePyramid(nn.Module):
    def __init__(self, conv_type, pyramid_feat, backbone_feat):
        super(MergePyramid, self).__init__()
        self.conv = conv_type(backbone_feat, pyramid_feat, kernel_size=1)

    def forward(self, previous_pyramid, backbone):
        return self.conv(backbone) + previous_pyramid

class StackedPyrHourGlass(nn.Module):
    def __init__(self,
                 nb_stacked,
                 in_channels,
                 pyramid_fmaps=128, 
                 stack_scaling_factors=((1,2,2), (1,2,2), (1,2,2)),
                 pyramidNet_kwargs=None,
                 patchNet_kwargs=None,
                 scale_spec_patchNet_kwargs=None,
                 trained_depths=None
                 ):
        super(StackedPyrHourGlass, self).__init__()
        self.nb_stacked = nb_stacked
        self.trained_depths = trained_depths

        # Build stacked pyramid models:
        from copy import deepcopy
        pyramidNet_kwargs["in_channels"] = in_channels + pyramid_fmaps
        pyramidNet_kwargs["pyramid_fmaps"] = pyramid_fmaps
        self.pyr_kwargs = [deepcopy(pyramidNet_kwargs) for _ in range(nb_stacked)]
        self.pyr_kwargs[0]["in_channels"] = in_channels
        self.pyr_models = nn.ModuleList([
            NewFeaturePyramidUNet3D(**kwrg) for kwrg in self.pyr_kwargs
        ])
        
        # Build patchNets:
        patchNet_kwargs["latent_variable_size"] = pyramid_fmaps
        self.ptch_kwargs = [deepcopy(patchNet_kwargs) for _ in range(nb_stacked)]
        if scale_spec_patchNet_kwargs is not None:
            assert len(scale_spec_patchNet_kwargs) == nb_stacked
            for i in range(nb_stacked):
                self.ptch_kwargs[i]["output_shape"] = scale_spec_patchNet_kwargs[i]["patch_size"]
                # self.ptch_kwargs[i].update(scale_spec_patchNet_kwargs[i])
        self.patch_models = nn.ModuleList([
            PatchNet(**kwgs) for kwgs in self.ptch_kwargs
        ])

        # Build crop-modules:
        assert len(stack_scaling_factors) == nb_stacked
        self.stack_scaling_factors = stack_scaling_factors
        from vaeAffs.transforms import DownsampleAndCrop3D
        self.crop_transforms = [DownsampleAndCrop3D(zoom_factor=(1, 1, 1),
                                                    crop_factor=scl_fact) for scl_fact in stack_scaling_factors]

        self.upsample_modules = nn.ModuleList([
            Upsample(scale_factor=tuple(scl_fact), mode="trilinear") for scl_fact in stack_scaling_factors
        ])

        self.fix_batchnorm_problem()

    def fix_batchnorm_problem(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, *inputs):
        LIMIT_STACK = 5
        
        assert len(inputs) == self.nb_stacked

        # Apply stacked models:
        current = None
        output_features = []
        i = 0
        for input, pyr_module, crop_transf, upsample in zip(inputs, self.pyr_models, self.crop_transforms,
                                                            self.upsample_modules):
            if current is None:
                current = input
            else:
                # Concatenate outputs to higher-res image:
                current = torch.cat((input, current), dim=1)
            pyramids = pyr_module(current)[:3] # Get only the highest pyramid
            # Save for loss:
            output_features += pyramids
            # output_features.append(pyramids)
            # Detach the gradient between pyramids for the moment:
            current = pyramids[0].detach()
            # Crop output:
            current = crop_transf.apply_to_torch_tensor(current)
            current = upsample(current)
            
            i += 1
            if i >= LIMIT_STACK:
                break
            
        return output_features

class CropVolumeModule(nn.Module):
    def __init__(self, scaling_factor):
        # TODO: this does not need to be a module...
        super(CropVolumeModule, self).__init__()
        assert len(scaling_factor) == 3
        self.scaling_factor = scaling_factor

    def forward(self, tensor):
        ndim = len(tensor.shape)
        crop_slc = tuple(slice(None) for _ in range(ndim-3)) + tuple(slice(crop[0], crop[1]) for crop in self.scaling_factor)
        return tensor[crop_slc]




class PatchNet(nn.Module):
    def __init__(self,
                 latent_variable_size=128,
                 output_shape=(5, 29, 29),
                 downscaling_factor=(1, 2, 2),
                 feature_maps=16):
        super(PatchNet, self).__init__()
        output_shape = tuple(output_shape) if isinstance(output_shape, list) else output_shape
        downscaling_factor = tuple(downscaling_factor) if isinstance(downscaling_factor, list) else downscaling_factor
        assert isinstance(downscaling_factor, tuple)

        output_shape = tuple(output_shape) if isinstance(output_shape, list) else output_shape
        assert isinstance(output_shape, tuple)
        self.output_shape = output_shape
        assert all(sh % 2 == 1 for sh in output_shape), "Patch should have even dimensions"

        self.min_path_shape = tuple(int(sh / dws) for sh, dws in zip(output_shape, downscaling_factor))
        self.vectorized_shape = np.array(self.min_path_shape).prod()

        # Build layers:
        self.linear_base = nn.Linear(latent_variable_size, self.vectorized_shape * feature_maps)

        self.upsampling = Upsample(scale_factor=downscaling_factor, mode="nearest")

        assert feature_maps % 2 == 0, "Necessary for group norm"
        self.decoder_module = ResBlockAdvanced(feature_maps, f_inner=feature_maps,
                                               f_out=1,
                                               dim=3,
                                               pre_kernel_size=(1, 3, 3),
                                               inner_kernel_size=(3, 3, 3),
                                               activation="ReLU",
                                               normalization="GroupNorm",
                                               num_groups_norm=2,
                                               apply_final_activation=False,
                                               apply_final_normalization=False)
        self.final_activation = nn.Sigmoid()

    def forward(self, encoded_variable):
        x = self.linear_base(encoded_variable)
        N = x.shape[0]
        reshaped = x.view(N, -1, *self.min_path_shape)

        # FIXME
        # upsampled = self.upsampling(reshaped)
        upsampled = reshaped

        # # Pad to correct shape:
        # padding = [[0,0], [0,0], [0,0]]
        # to_be_padded = False
        # for d in range(3):
        #     diff = self.output_shape[d] - upsampled.shape[d - 3]
        #     if diff != 0:
        #         padding[d][0] = diff
        #         to_be_padded = True
        # if to_be_padded:
        #     padding.reverse() # Pytorch expect the opposite order
        #     padding = [tuple(pad) for pad in padding]
        #     upsampled = nn.functional.pad(upsampled, padding[0]+padding[1]+padding[2], mode='replicate')

        conved = self.decoder_module(upsampled)

        return self.final_activation(conved)






class FeaturePyramidUNet3D(EncoderDecoderSkeleton):
    def __init__(self, depth, in_channels, encoder_fmaps, pyramid_fmaps,
                 AE_kwargs=None,
                 scale_factor=2,
                 conv_type='vanilla',
                 final_activation=None,
                 upsampling_mode='nearest',
                 **kwargs):
        # TODO: improve this crap
        self.depth = depth
        self.in_channels = in_channels
        self.pyramid_fmaps = pyramid_fmaps

        if isinstance(encoder_fmaps, (list, tuple)):
            self.encoder_fmaps = encoder_fmaps
        else:
            assert isinstance(encoder_fmaps, int)
            if 'fmap_increase' in kwargs:
                self.fmap_increase = kwargs['fmap_increase']
                self.encoder_fmaps = [encoder_fmaps + i * self.fmap_increase for i in range(self.depth + 1)]
            elif 'fmap_factor' in kwargs:
                self.fmap_factor = kwargs['fmap_factor']
                self.encoder_fmaps = [encoder_fmaps * self.fmap_factor**i for i in range(self.depth + 1)]
            else:
                self.encoder_fmaps = [encoder_fmaps, ] * (self.depth + 1)

        self.final_activation = [final_activation] if final_activation is not None else None

        # parse conv_type
        if isinstance(conv_type, str):
            assert conv_type in CONV_TYPES
            self.conv_type = CONV_TYPES[conv_type]
        else:
            assert isinstance(conv_type, type)
            self.conv_type = conv_type

        # parse scale factor
        if isinstance(scale_factor, int):
            scale_factor = [scale_factor, ] * depth
        scale_factors = scale_factor
        normalized_factors = []
        for scale_factor in scale_factors:
            assert isinstance(scale_factor, (int, list, tuple))
            if isinstance(scale_factor, int):
                scale_factor = self.dim * [scale_factor, ]
            assert len(scale_factor) == self.dim
            normalized_factors.append(scale_factor)
        self.scale_factors = normalized_factors
        self.upsampling_mode = upsampling_mode

        # compute input size divisibiliy constraints
        divisibility_constraint = np.ones(len(self.scale_factors[0]))
        for scale_factor in self.scale_factors:
            divisibility_constraint *= np.array(scale_factor)
        self.divisibility_constraint = list(divisibility_constraint.astype(int))



        super(FeaturePyramidUNet3D, self).__init__(depth)

        # self.shortcut_convs = nn.ModuleList([self.build_shortcut_conv(d) for d in range(1,depth)])
        # self.shortcut_merge = nn.ModuleList([self.build_shortcut_merge(d) for d in range(1, depth)])
        #
        # # Convolution producing affinities:
        # self.final_conv = nn.Sequential(self.construct_conv(self.pyramid_fmaps*3 + 3, self.pyramid_fmaps, kernel_size=1),
        #                                 self.construct_conv(self.pyramid_fmaps, 1, kernel_size=1))
        # self.sigmoid = nn.Sigmoid()

        from vaeAffs.models.vanilla_vae import AutoEncoder
        self.AE_model = nn.ModuleList([AutoEncoder(**AE_kwargs)  for _ in range(3)])

        # # Load final decoders:
        # assert isinstance(path_autoencoder_model, str)
        # # FIXME: this should be moved to the model, otherwise it's not saved!
        # self.AE_model = [torch.load(path_autoencoder_model),
        #                    torch.load(path_autoencoder_model),
        #                    torch.load(path_autoencoder_model),]

        for i in range(3):
            self.AE_model[i].set_min_patch_shape(tuple(AE_kwargs.get("patch_size")))

            # Freeze the auto-encoder model:
            # for param in self.AE_model[i].parameters():
            #     param.requires_grad = False

        self.fix_batchnorm_problem()

    def fix_batchnorm_problem(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def build_shortcut_conv(self, depth):
        # build final bottom-up shortcut:
        from inferno.extensions.layers import ConvActivation
        from inferno.extensions.initializers.presets import OrthogonalWeightsZeroBias
        return ConvActivation(self.pyramid_fmaps, self.pyramid_fmaps, (1,3,3), dim=3,
                       stride=self.scale_factors[depth],
                       dilation=(1,3,3),
                       activation='ELU',
                       initialization=OrthogonalWeightsZeroBias())

    def build_shortcut_merge(self, depth):
        return Sum()

    def forward(self, *inputs):
        input, offsets = inputs
        encoded_states = []
        current = input
        for encode, downsample in zip(self.encoder_modules, self.downsampling_modules):
            current = encode(current)
            encoded_states.append(current)
            current = downsample(current)
        current = self.base_module(current)

        feature_pyramid = [current]
        for encoded_state, upsample, skip, merge, decode, depth in reversed(list(zip(
                encoded_states, self.upsampling_modules, self.skip_modules, self.merge_modules, self.decoder_modules, range(len(self.decoder_modules))))):
            # if depth == 0:
            #     break
            current = upsample(current)
            current = merge(current, encoded_state)
            current = decode(current)
            feature_pyramid.append(current)

        feature_pyramid.reverse()
        # Bottom-up shortcut:
        # feature_pyramid_upscaled = [current]
        # for d, conv, merge  in zip(range(0, self.depth-1), self.shortcut_convs, self.shortcut_merge):
        #     current = conv(current)
        #     current = merge(current, feature_pyramid[d])
        #     # feature_pyramid[d] = current
        # #     # Upscale:
        # #     if d == 1:
        # #         feature_pyramid_upscaled.append(self.upsampling_modules[1](current))
        # #     elif d == 2:
        # #         feature_pyramid_upscaled.append(self.upsampling_modules[1](self.upsampling_modules[2](current)))
        # #     else:
        # #         raise NotImplementedError("Temp hack, only working with depth 3")
        # for d in range(1, self.depth):
        #     # Upscale:
        #     if d == 1:
        #         feature_pyramid_upscaled.append(self.upsampling_modules[1](feature_pyramid[d]))
        #     elif d == 2:
        #         feature_pyramid_upscaled.append(self.upsampling_modules[1](self.upsampling_modules[2](feature_pyramid[d])))
        #     else:
        #         raise NotImplementedError("Temp hack, only working with depth 3")



        # # Perform final convolution:
        # concatenated_features = torch.cat(tuple(feature_pyramid_upscaled)+(self.downsampling_modules[0](offsets.float()),), 1)
        # affinities = self.sigmoid(self.final_conv(concatenated_features))
        # return [affinities] + feature_pyramid
        return feature_pyramid

    def construct_merge_module(self, depth):
        return MergePyramid(self.conv_type, self.pyramid_fmaps, self.encoder_fmaps[depth])

    def construct_encoder_module(self, depth):
        f_in = self.encoder_fmaps[depth - 1] if depth != 0 else self.in_channels
        f_out = self.encoder_fmaps[depth]
        if depth != 0:
            return nn.Sequential(
                ResBlockAdvanced(f_in, f_inner=f_out, pre_kernel_size=(1, 3, 3),
                                 inner_kernel_size=(3, 3, 3),
                                 activation="ReLU",
                                 normalization="GroupNorm",
                                 num_groups_norm=16,  # TODO: generalize
                                 ),
                ResBlockAdvanced(f_out, f_inner=f_out, pre_kernel_size=(1, 3, 3),
                                 inner_kernel_size=(3, 3, 3),
                                 activation="ReLU",
                                 normalization="GroupNorm",
                                 num_groups_norm=16,  # TODO: generalize
                                 ),
                # ResBlockAdvanced(f_out, f_inner=f_out, pre_kernel_size=(1, 3, 3),
                #                  inner_kernel_size=(3, 3, 3),
                #                  activation="ReLU",
                #                  normalization="GroupNorm",
                #                  num_groups_norm=15,  # TODO: generalize
                #                  ),
            )
        if depth == 0:
            return ResBlockAdvanced(f_in, f_inner=f_out, pre_kernel_size=(1, 5, 5),
                                 inner_kernel_size=(1, 3, 3),
                                 activation="ReLU",
                                 normalization="GroupNorm",
                                 num_groups_norm=10,  # TODO: generalize
                                 )

    def construct_decoder_module(self, depth):
        return ResBlockAdvanced(self.pyramid_fmaps, f_inner=self.pyramid_fmaps, 
                                pre_kernel_size=(1, 1, 1),
                                 inner_kernel_size=(3, 3, 3),
                                 activation="ReLU",
                                 normalization="GroupNorm",
                                 num_groups_norm=16,  # TODO: generalize
                                 )

    def construct_base_module(self):
        f_in = self.encoder_fmaps[self.depth - 1]
        f_intermediate = self.encoder_fmaps[self.depth]
        f_out = self.pyramid_fmaps
        return ResBlockAdvanced(f_in, f_inner=f_intermediate,
                                f_out=f_out,
                                pre_kernel_size=(1, 3, 3),
                                 inner_kernel_size=(3, 3, 3),
                                 activation="ReLU",
                                 normalization="GroupNorm",
                                 num_groups_norm=16,  # TODO: generalize
                                 )
    @property
    def dim(self):
        return 3

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
        sampler = Upsample(scale_factor=scale_factor, mode=self.upsampling_mode)
        return sampler

    def construct_conv(self, f_in, f_out, kernel_size=3):
        return self.conv_type(f_in, f_out, kernel_size=kernel_size)

class NewFeaturePyramidUNet3D(FeaturePyramidUNet3D):
    def __init__(self, depth, in_channels, encoder_fmaps, pyramid_fmaps,
                 scale_factor=2,
                 conv_type='vanilla',
                 final_activation=None,
                 upsampling_mode='nearest',
                 **kwargs):
        # TODO: improve this crap
        self.depth = depth
        self.in_channels = in_channels
        self.pyramid_fmaps = pyramid_fmaps

        if isinstance(encoder_fmaps, (list, tuple)):
            self.encoder_fmaps = encoder_fmaps
        else:
            assert isinstance(encoder_fmaps, int)
            if 'fmap_increase' in kwargs:
                self.fmap_increase = kwargs['fmap_increase']
                self.encoder_fmaps = [encoder_fmaps + i * self.fmap_increase for i in range(self.depth + 1)]
            elif 'fmap_factor' in kwargs:
                self.fmap_factor = kwargs['fmap_factor']
                self.encoder_fmaps = [encoder_fmaps * self.fmap_factor ** i for i in range(self.depth + 1)]
            else:
                self.encoder_fmaps = [encoder_fmaps, ] * (self.depth + 1)

        self.final_activation = [final_activation] if final_activation is not None else None

        # parse conv_type
        if isinstance(conv_type, str):
            assert conv_type in CONV_TYPES
            self.conv_type = CONV_TYPES[conv_type]
        else:
            assert isinstance(conv_type, type)
            self.conv_type = conv_type

        # parse scale factor
        if isinstance(scale_factor, int):
            scale_factor = [scale_factor, ] * depth
        scale_factors = scale_factor
        normalized_factors = []
        for scale_factor in scale_factors:
            assert isinstance(scale_factor, (int, list, tuple))
            if isinstance(scale_factor, int):
                scale_factor = self.dim * [scale_factor, ]
            assert len(scale_factor) == self.dim
            normalized_factors.append(scale_factor)
        self.scale_factors = normalized_factors
        self.upsampling_mode = upsampling_mode

        # compute input size divisibiliy constraints
        divisibility_constraint = np.ones(len(self.scale_factors[0]))
        for scale_factor in self.scale_factors:
            divisibility_constraint *= np.array(scale_factor)
        self.divisibility_constraint = list(divisibility_constraint.astype(int))

        super(FeaturePyramidUNet3D, self).__init__(depth)


    def forward(self, input):
        return super(NewFeaturePyramidUNet3D, self).forward(*(input, input))


    def construct_encoder_module(self, depth):
        f_in = self.encoder_fmaps[depth - 1] if depth != 0 else self.in_channels
        f_out = self.encoder_fmaps[depth]
        if depth != 0:
            return nn.Sequential(
                ResBlockAdvanced(f_in, f_inner=f_out, pre_kernel_size=(1, 3, 3),
                                 inner_kernel_size=(3, 3, 3),
                                 activation="ReLU",
                                 normalization="GroupNorm",
                                 num_groups_norm=16,  # TODO: generalize
                                 ),
            )
        if depth == 0:
            return ResBlockAdvanced(f_in, f_inner=f_out, pre_kernel_size=(1, 5, 5),
                                 inner_kernel_size=(1, 3, 3),
                                 activation="ReLU",
                                 normalization="GroupNorm",
                                 num_groups_norm=16,  # TODO: generalize
                                 )

    def construct_decoder_module(self, depth):
        return ResBlockAdvanced(self.pyramid_fmaps, f_inner=self.pyramid_fmaps,
                                pre_kernel_size=(1, 1, 1),
                                 inner_kernel_size=(3, 3, 3),
                                 activation="ReLU",
                                 normalization="GroupNorm",
                                 num_groups_norm=16,  # TODO: generalize
                                 )

    def construct_base_module(self):
        f_in = self.encoder_fmaps[self.depth - 1]
        f_intermediate = self.encoder_fmaps[self.depth]
        f_out = self.pyramid_fmaps
        return ResBlockAdvanced(f_in, f_inner=f_intermediate,
                                f_out=f_out,
                                pre_kernel_size=(1, 3, 3),
                                 inner_kernel_size=(3, 3, 3),
                                 activation="ReLU",
                                 normalization="GroupNorm",
                                 num_groups_norm=16,  # TODO: generalize
                                 )

class UNetSkeleton(EncoderDecoderSkeleton):

    def __init__(self, depth, in_channels, out_channels, fmaps, **kwargs):
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(fmaps, (list, tuple)):
            self.fmaps = fmaps
        else:
            assert isinstance(fmaps, int)
            if 'fmap_increase' in kwargs:
                self.fmap_increase = kwargs['fmap_increase']
                self.fmaps = [fmaps + i * self.fmap_increase for i in range(self.depth + 1)]
            elif 'fmap_factor' in kwargs:
                self.fmap_factor = kwargs['fmap_factor']
                self.fmaps = [fmaps * self.fmap_factor**i for i in range(self.depth + 1)]
            else:
                self.fmaps = [fmaps, ] * (self.depth + 1)
        assert len(self.fmaps) == self.depth + 1

        self.merged_fmaps = [2 * n for n in self.fmaps]

        super(UNetSkeleton, self).__init__(depth)

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
        f_in = self.fmaps[self.depth - 1]
        f_intermediate = self.fmaps[self.depth]
        f_out = self.fmaps[self.depth - 1]
        return nn.Sequential(
            self.construct_conv(f_in, f_intermediate),
            self.construct_conv(f_intermediate, f_out)
        )


CONV_TYPES = {'vanilla': ConvELU3D,
              'conv_bn': BNReLUConv3D,
              'vanilla2D': ConvELU2D,
              'conv_bn2D': BNReLUConv2D}


class UNet3D(UNetSkeleton):

    def __init__(self,
                 scale_factor=2,
                 conv_type='vanilla',
                 final_activation=None,
                 upsampling_mode='nearest',
                 *super_args, **super_kwargs):


        self.final_activation = [final_activation] if final_activation is not None else None

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
                scale_factor = self.dim * [scale_factor, ]
            assert len(scale_factor) == self.dim
            normalized_factors.append(scale_factor)
        self.scale_factors = normalized_factors
        self.upsampling_mode = upsampling_mode

        # compute input size divisibiliy constraints
        divisibility_constraint = np.ones(len(self.scale_factors[0]))
        for scale_factor in self.scale_factors:
            divisibility_constraint *= np.array(scale_factor)
        self.divisibility_constraint = list(divisibility_constraint.astype(int))

        super(UNet3D, self).__init__(*super_args, **super_kwargs)
        # self.setup_graph()  # TODO: this is ugly. do it when forward() is called for the first time?

    def construct_conv(self, f_in, f_out, kernel_size=3):
        return self.conv_type(f_in, f_out, kernel_size=kernel_size)

    def construct_output_module(self):
        if self.final_activation is not None:
            return self.final_activation[0]
        else:
            return Identity()

    @property
    def dim(self):
        return 3

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
        sampler = Upsample(scale_factor=scale_factor, mode=self.upsampling_mode)
        return sampler

    def forward(self, input_):
        # input_dim = len(input_.shape)
        # assert all(input_.shape[-i] % self.divisibility_constraint[-i] == 0 for i in range(1, input_dim-1)), \
            # f'Volume dimensions {input_.shape[2:]} are not divisible by {self.divisibility_constraint}'
        return super(UNet3D, self).forward(input_)


class SuperhumanSNEMINet(UNet3D):
    # see https://arxiv.org/pdf/1706.00120.pdf

    def __init__(self,
                 in_channels=1, out_channels=1,
                 fmaps=(28, 36, 48, 64, 80),
                 conv_type=ConvELU3D,
                 scale_factor=(
                     (1, 2, 2),
                     (1, 2, 2),
                     (1, 2, 2),
                     (1, 2, 2)
                 ),
                 depth=None,
                 **kwargs):
        if depth is None:
            depth = len(fmaps) - 1
        super(SuperhumanSNEMINet, self).__init__(
            conv_type=conv_type,
            depth=depth,
            fmaps=fmaps,
            in_channels=in_channels,
            out_channels=out_channels,
            scale_factor=scale_factor,
            **kwargs
        )

    def construct_merge_module(self, depth):
        return Sum()

    def construct_encoder_module(self, depth):
        f_in = self.fmaps[depth - 1] if depth != 0 else self.in_channels
        f_out = self.fmaps[depth]
        if depth != 0:
            return SuperhumanSNEMIBlock(f_in=f_in, f_out=f_out, conv_type=self.conv_type)
        if depth == 0:
            return SuperhumanSNEMIBlock(f_in=f_in, f_out=f_out, conv_type=self.conv_type,
                                        pre_kernel_size=(1, 5, 5), inner_kernel_size=(1, 3, 3))

    def construct_decoder_module(self, depth):
        f_in = self.fmaps[depth]
        f_out = self.fmaps[0] if depth == 0 else self.fmaps[depth - 1]
        if depth != 0:
            return SuperhumanSNEMIBlock(f_in=f_in, f_out=f_out, conv_type=self.conv_type)
        if depth == 0:
            return nn.Sequential(
                SuperhumanSNEMIBlock(f_in=f_in, f_out=f_out, conv_type=self.conv_type,
                                     pre_kernel_size=(3, 3, 3), inner_kernel_size=(1, 3, 3)),
                self.conv_type(f_out, self.out_channels, kernel_size=(1, 5, 5))
            )

    def construct_base_module(self):
        f_in = self.fmaps[self.depth - 1]
        f_intermediate = self.fmaps[self.depth]
        f_out = self.fmaps[self.depth - 1]
        return SuperhumanSNEMIBlock(f_in=f_in, f_main=f_intermediate, f_out=f_out, conv_type=self.conv_type)


class AffinityNet(nn.Module):
    def __init__(self, path_PyrUNet, nb_offsets, *super_args, **super_kwargs):
        super(AffinityNet, self).__init__()
        
        self.pyr_unet_model = torch.load(path_PyrUNet)["_model"]

        # # Freeze the network parameters:
        # for param in self.pyr_unet_model.parameters():
        #     param.requires_grad = False

        nb_pyr_maps = self.pyr_unet_model.pyramid_fmaps
        
        self.final_module = nn.Sequential(
            ResBlockAdvanced(f_in=nb_pyr_maps*3, f_out=nb_pyr_maps,pre_kernel_size=(1,3,3),
                             inner_kernel_size=(1,3,3),num_groups_norm=16),
            ResBlockAdvanced(f_in=nb_pyr_maps, f_out=nb_pyr_maps, pre_kernel_size=(1, 3, 3),
                             inner_kernel_size=(3, 3, 3), num_groups_norm=16, dilation=3),
            ResBlockAdvanced(f_in=nb_pyr_maps, f_out=nb_offsets, pre_kernel_size=(1, 3, 3),
                             inner_kernel_size=(3, 3, 3), num_groups_norm=16, dilation=4,
                             apply_final_activation=False,
                             apply_final_normalization=False),
            # SuperhumanSNEMIBlock(f_in=nb_pyr_maps*3, f_out=nb_pyr_maps,
            #                                      conv_type=self.pyr_unet_model.conv_type),
            # SuperhumanSNEMIBlock(f_in=nb_pyr_maps, f_out=nb_pyr_maps,
            #                      conv_type=self.pyr_unet_model.conv_type, dilation=3),
            # SuperhumanSNEMIBlock(f_in=nb_pyr_maps, f_out=nb_offsets,
            #                      conv_type=self.pyr_unet_model.conv_type, dilation=4),
        )

        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, raw):
        # Hack because this model was trained with affs:
        with torch.no_grad():
            feat_pyr = self.pyr_unet_model.forward(*(raw, raw))

        upscaled_feat_pyr = feat_pyr
        upscaled_feat_pyr[1] = self.pyr_unet_model.upsampling_modules[1](feat_pyr[1])
        upscaled_feat_pyr[2] = self.pyr_unet_model.upsampling_modules[1](
            self.pyr_unet_model.upsampling_modules[2](feat_pyr[2]))

        output = self.final_module(torch.cat(tuple(upscaled_feat_pyr[:3]), dim=1))
        return self.sigmoid(output)



class MaskUNet(UNet3D):
    def __init__(self, path_PyrUNet, *super_args, **super_kwargs):
        pyr_unet_model = torch.load(path_PyrUNet)["_model"]
        nb_pyr_maps = pyr_unet_model.pyramid_fmaps
        
        super_kwargs["in_channels"] = nb_pyr_maps*3 + super_kwargs.get("out_channels", 1)
        super_kwargs["out_channels"] = super_kwargs.get("out_channels", 1)
        super_kwargs["final_activation"] = nn.Sigmoid()
        
        super(MaskUNet, self).__init__(*super_args, **super_kwargs)

        self.pyr_unet_model = pyr_unet_model


    def forward(self, *inputs):
        raw, mask = inputs
        
        # Hack because this model was trained with affs:
        with torch.no_grad():
            feat_pyr = self.pyr_unet_model.forward(*(raw, raw))

        # Upscale:
        upscaled_feat_pyr = feat_pyr
        upscaled_feat_pyr[1] = self.pyr_unet_model.upsampling_modules[1](feat_pyr[1])
        upscaled_feat_pyr[2] = self.pyr_unet_model.upsampling_modules[1](
            self.pyr_unet_model.upsampling_modules[2](feat_pyr[2]))

        
        out = super(MaskUNet, self).forward(torch.cat(tuple(upscaled_feat_pyr[:3]) + (mask,), dim=1))
        out = super(MaskUNet, self).forward(torch.cat(tuple(upscaled_feat_pyr[:3]) + (out,), dim=1))
        # out = super(MaskUNet, self).forward(torch.cat(tuple(upscaled_feat_pyr[:3]) + (out,), dim=1))

        return out

class ShakeShakeSNEMINet(SuperhumanSNEMINet):

    def construct_merge_module(self, depth):
        return ShakeShakeMerge()


class UNet2D(UNet3D):

    def __init__(self,
                 conv_type='vanilla2D',
                 *super_args, **super_kwargs):
        super_kwargs.update({"conv_type": conv_type})
        super(UNet2D, self).__init__(*super_args, **super_kwargs)

    @property
    def dim(self):
        return 2

    def construct_downsampling_module(self, depth):
        scale_factor = self.scale_factors[depth]
        sampler = nn.MaxPool2d(kernel_size=scale_factor,
                               stride=scale_factor,
                               padding=0)
        return sampler


class RecurrentUNet2D(UNet2D):

    def __init__(self,
                 scale_factor=2,
                 conv_type='vanilla2D',
                 final_activation=None,
                 rec_n_layers=3,
                 rec_kernel_sizes=(3, 5, 3),
                 rec_hidden_size=(16, 32, 8),
                 *super_args, **super_kwargs):
        self.rec_n_layers = rec_n_layers
        self.rec_kernel_sizes = rec_kernel_sizes
        self.rec_hidden_size = rec_hidden_size
        self.rec_layers = []

        super(UNet2D, self).__init__(scale_factor=scale_factor,
                                     conv_type=conv_type,
                                     final_activation=final_activation,
                                     *super_args, **super_kwargs)

    def construct_skip_module(self, depth):
        self.rec_layers.append(ConvGRU(input_size=self.fmaps[depth],
                                       hidden_size=self.fmaps[depth],
                                       kernel_sizes=self.rec_kernel_sizes,
                                       n_layers=self.rec_n_layers,
                                       conv_type=self.conv_type))

        return self.rec_layers[-1]

    def set_sequence_length(self, sequence_length):
        for r in self.rec_layers:
            r.sequence_length = sequence_length

    def forward(self, input_):
        sequence_length = input_.shape[2]
        batch_size = input_.shape[0]
        flat_shape = [batch_size * sequence_length] + [input_.shape[1]] + list(input_.shape[3:])
        flat_output_shape = [batch_size, sequence_length] + [self.out_channels] + list(input_.shape[3:])

        transpose_input = input_.permute((0, 2, 1, 3, 4))\
                                .contiguous()\
                                .view(*flat_shape).detach()

        self.set_sequence_length(sequence_length)
        output = super(UNet2D, self).forward(transpose_input)
        return output.view(*flat_output_shape).permute((0, 2, 1, 3, 4))


if __name__ == '__main__':
    model = SuperhumanSNEMINet(
        in_channels=1,
        out_channels=2,
        final_activation=nn.Sigmoid()
    )

    print(model)
    model = model.cuda()
    inp = torch.ones(tuple((np.array(model.divisibility_constraint) * 2)))[None, None].cuda()
    out = model(inp)
    print(inp.shape)
    print(out.shape)
