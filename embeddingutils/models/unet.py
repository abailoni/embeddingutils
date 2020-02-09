from inferno.extensions.containers.graph import Graph, Identity

from inferno.extensions.layers.convolutional import ConvELU3D, Conv3D, BNReLUConv3D
from inferno.extensions.layers.convolutional import ConvELU2D, Conv2D, BNReLUConv2D
from inferno.extensions.layers.sampling import AnisotropicPool, AnisotropicUpsample
from inferno.extensions.layers.reshape import Concatenate, Sum

from embeddingutils.models.submodules import SuperhumanSNEMIBlock, ConvGRU, ShakeShakeMerge, Upsample
from embeddingutils.models.submodules import ResBlockAdvanced

import torch
import torch.nn as nn
from copy import deepcopy
from inferno.extensions.layers.convolutional import ConvNormActivation

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

class MergePyramidNew(nn.Module):
    def __init__(self, pyramid_feat, backbone_feat):
        super(MergePyramidNew, self).__init__()
        self.conv = ConvNormActivation(backbone_feat, pyramid_feat, kernel_size=(1, 1, 1),
                                  dim=3,
                                       activation="ReLU",
                                       normalization="GroupNorm",
                                       num_groups_norm=16)

    def forward(self, previous_pyramid, backbone):
        return self.conv(backbone) + previous_pyramid

class UpsampleAndCrop(nn.Module):
    def __init__(self, scale_factor, mode,
                                  crop_slice=None):
        super(UpsampleAndCrop, self).__init__()
        self.upsampler = Upsample(scale_factor=scale_factor, mode=mode)
        self.crop_slice = crop_slice

        if self.crop_slice is not None:
            assert isinstance(self.crop_slice, str)
            from inferno.io.volumetric.volumetric_utils import parse_data_slice
            self.crop_slice = (slice(None), slice(None)) + parse_data_slice(self.crop_slice)

    def forward(self, input):
        if isinstance(input, tuple):
            input = input[0]
        # print(input.shape)
        if self.crop_slice is not None:
            input = input[self.crop_slice]
        output = self.upsampler(input)
        return output

class Crop(nn.Module):
    def __init__(self, crop_slice):
        super(Crop, self).__init__()
        self.crop_slice = crop_slice

        if self.crop_slice is not None:
            assert isinstance(self.crop_slice, str)
            from inferno.io.volumetric.volumetric_utils import parse_data_slice
            self.crop_slice = (slice(None), slice(None)) + parse_data_slice(self.crop_slice)

    def forward(self, input):
        if isinstance(input, tuple):
            raise NotImplementedError("At the moment only one input is accepted")
        if self.crop_slice is not None:
            return input[self.crop_slice]
        else:
            return input



class MergePyramidAndAutoCrop(nn.Module):
    def __init__(self, pyramid_feat, backbone_feat):
        super(MergePyramidAndAutoCrop, self).__init__()
        if pyramid_feat == backbone_feat:
            self.conv = Identity()
        else:
            self.conv = ConvNormActivation(backbone_feat, pyramid_feat, kernel_size=(1, 1, 1),
                                  dim=3,
                                       activation="ReLU",
                                       normalization="GroupNorm",
                                       num_groups_norm=16)


    def forward(self, previous_pyramid, backbone):
        if previous_pyramid.shape[2:] != backbone.shape[2:]:
            target_shape = previous_pyramid.shape[2:]
            orig_shape = backbone.shape[2:]
            diff = [orig-trg for orig, trg in zip(orig_shape, target_shape)]
            crop_backbone = True
            if not all([d>=0 for d in diff]):
                crop_backbone = False
                orig_shape, target_shape = target_shape, orig_shape
                diff = [orig - trg for orig, trg in zip(orig_shape, target_shape)]
            left_crops = [int(d/2) for d in diff]
            right_crops = [shp-int(d/2) if d%2==0 else shp-(int(d/2)+1)  for d, shp in zip(diff, orig_shape)]
            crop_slice = (slice(None), slice(None)) + tuple(slice(lft,rgt) for rgt,lft in zip(right_crops, left_crops))
            if crop_backbone:
                backbone = backbone[crop_slice]
            else:
                previous_pyramid = previous_pyramid[crop_slice]

        return self.conv(backbone) + previous_pyramid


class AutoPad(nn.Module):
    def __init__(self):
        super(AutoPad, self).__init__()

    def forward(self, to_be_padded, out_shape):
        in_shape = to_be_padded.shape[2:]
        out_shape = out_shape[2:]
        if in_shape != out_shape:
            diff = [trg-orig for orig, trg in zip(in_shape, out_shape)]
            assert all([d>=0 for d in diff]), "Output shape should be bigger"
            assert all([d % 2 == 0 for d in diff]), "Odd difference in shape!"
            # F.pad expects the last dim first:
            diff.reverse()
            pad = []
            for d in diff:
                pad += [int(d/2), int(d/2)]
            to_be_padded = torch.nn.functional.pad(to_be_padded, tuple(pad), mode='constant', value=0)
        return to_be_padded


def auto_crop_tensor_to_shape(to_be_cropped, target_tensor_shape, return_slice=False,
                              ignore_channel_and_batch_dims=True):
    initial_shape = to_be_cropped.shape
    diff = [int_sh - trg_sh for int_sh, trg_sh in zip(initial_shape, target_tensor_shape)]
    if ignore_channel_and_batch_dims:
        assert all([d >= 0 for d in diff[2:]]), "Target shape should be smaller!"
    else:
        assert all([d >= 0 for d in diff]), "Target shape should be smaller!"
    left_crops = [int(d / 2) for d in diff]
    right_crops = [shp - int(d / 2) if d % 2 == 0 else shp - (int(d / 2) + 1) for d, shp in zip(diff, initial_shape)]
    if ignore_channel_and_batch_dims:
        crop_slice = (slice(None), slice(None)) + tuple(slice(lft, rgt) for rgt, lft in zip(right_crops[2:], left_crops[2:]))
    else:
        crop_slice = tuple(slice(lft, rgt) for rgt, lft in zip(right_crops, left_crops))
    if return_slice:
        return crop_slice
    else:
        return to_be_cropped[crop_slice]


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

    def crop_and_upsample(self, tensor, stack_depth):
        tensor = self.crop_transforms[stack_depth].apply_to_torch_tensor(tensor)
        return self.upsample_modules[stack_depth](tensor)

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


class PatchNetVAE(nn.Module):
    def __init__(self, patchNet_kwargs, pre_maxpool=None):
        super(PatchNetVAE, self).__init__()
        assert patchNet_kwargs['downscaling_factor'] == (1, 1, 1) or patchNet_kwargs['downscaling_factor'] == [1, 1, 1], "Not implemented atm"
        self.patch_net = ptch = PatchNet(**patchNet_kwargs)

        # Build the encoder:
        self.encoder_module = ResBlockAdvanced(1, f_inner=ptch.feature_maps,
                                               f_out=ptch.feature_maps,
                                               dim=3,
                                               pre_kernel_size=(1, 3, 3),
                                               inner_kernel_size=(3, 3, 3),
                                               activation="ReLU",
                                               normalization="GroupNorm",
                                               num_groups_norm=2,)
        self.linear_encoder = nn.Linear(ptch.vectorized_shape * ptch.feature_maps, ptch.latent_variable_size * 2)

        self.pre_maxpool = None
        if pre_maxpool is not None:
            self.pre_maxpool = nn.MaxPool3d(kernel_size=pre_maxpool,
                                            stride=pre_maxpool,
                                            padding=0)

    def encoder(self, input):
        N = input.shape[0]
        encoded = self.encoder_module(input)
        reshaped = encoded.view(N, -1)
        vector = self.linear_encoder(reshaped)
        latent_var_size = self.patch_net.latent_variable_size
        return vector[:,:latent_var_size], vector[:,latent_var_size:]

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def generate(self, shape):
        return torch.randn(shape)


    def forward(self, input):
        if self.pre_maxpool is not None:
            input = self.pre_maxpool(input)

        # encoded_variable = self.encode(input_)
        mu, logvar = self.encoder(input)

        z = self.reparameterize(mu, logvar)

        reconstructed = self.patch_net(z)

        # Expand to avoid complains from firelight:
        mu = mu.unsqueeze(-1)
        mu = mu.unsqueeze(-1)
        mu = mu.unsqueeze(-1)
        logvar = logvar.unsqueeze(-1)
        logvar = logvar.unsqueeze(-1)
        logvar = logvar.unsqueeze(-1)

        return [reconstructed, mu, logvar]


class PatchNet(nn.Module):
    def __init__(self,
                 latent_variable_size=128,
                 output_shape=(5, 29, 29),
                 downscaling_factor=(1, 2, 2),
                 feature_maps=16,
                 legacy=False,
                 legacy_resblock=None,
                 **extra_kwargs):
        super(PatchNet, self).__init__()

        # FIXME: ugly hack
        ptch_size = extra_kwargs.get("patch_size")
        output_shape = ptch_size if ptch_size is not None else output_shape

        assert downscaling_factor == (1,1,1) or downscaling_factor == [1,1,1], "Not implemented atm"

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
        self.latent_variable_size = latent_variable_size
        self.linear_base = nn.Linear(latent_variable_size, self.vectorized_shape * feature_maps)

        self.upsampling = Upsample(scale_factor=downscaling_factor, mode="nearest")
        # self.norm = nn.GroupNorm(num_channels=self.vectorized_shape * feature_maps, num_groups=feature_maps)
        # self.relu = nn.ReLU()

        # Legacy:
        pre_kernel_size = (3, 3, 3)
        num_groups_norm = 1
        if legacy:
            pre_kernel_size = (1, 3, 3)
            num_groups_norm = 2

        # assert feature_maps % 2 == 0, "Necessary for group norm"
        self.feature_maps = feature_maps
        self.decoder_module = ResBlockAdvanced(feature_maps, f_inner=feature_maps,
                                               f_out=1,
                                               dim=3,
                                               pre_kernel_size=pre_kernel_size,
                                               inner_kernel_size=(3, 3, 3),
                                               activation="ReLU",
                                               normalization="GroupNorm",
                                               num_groups_norm=num_groups_norm,
                                               legacy_version=legacy_resblock,
                                               apply_final_activation=False,
                                               apply_final_normalization=False)
        self.final_activation = nn.Sigmoid()

    def forward(self, encoded_variable):
        x = self.linear_base(encoded_variable)
        # x = self.norm(x)
        # x = self.relu(x)
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

        out = self.decoder_module(upsampled)
        out = self.final_activation(out)
        return out






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
        # TODO: initialize residual blocks in the smart way?
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # def build_shortcut_conv(self, depth):
    #     # build final bottom-up shortcut:
    #     from inferno.extensions.layers import ConvActivation
    #     from inferno.extensions.initializers.presets import OrthogonalWeightsZeroBias
    #     return ConvActivation(self.pyramid_fmaps, self.pyramid_fmaps, (1,3,3), dim=3,
    #                    stride=self.scale_factors[depth],
    #                    dilation=(1,3,3),
    #                    activation='ELU',
    #                    initialization=OrthogonalWeightsZeroBias())
    #
    # def build_shortcut_merge(self, depth):
    #     return Sum()

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


class YetAnotherFeaturePyramidUNet3D(FeaturePyramidUNet3D):
    def __init__(self, depth, in_channels, encoder_fmaps, pyramid_fmaps,
                 scale_factor=2,
                 conv_type='vanilla',
                 final_activation=None,
                 upsampling_mode='nearest',
                 patchNet_kwargs=None,
                 scale_spec_patchNet_kwargs=None,
                 output_fmaps=None,
                 stop_decoder_at_depth=0, # Sometimes we stop the decoder earlier
                 nb_patch_nets=2,
                 **kwargs):
        # TODO: improve this crap
        self.depth = depth
        self.in_channels = in_channels
        self.pyramid_fmaps = pyramid_fmaps
        self.output_fmaps = pyramid_fmaps if output_fmaps is None else output_fmaps
        self.stop_decoder_at_depth = stop_decoder_at_depth


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

        # Build patchNets:
        # patchNet_kwargs["latent_variable_size"] = pyramid_fmaps
        self.ptch_kwargs = [deepcopy(patchNet_kwargs) for _ in range(nb_patch_nets)]
        # self.scale_spec_patchNet_kwargs = scale_spec_patchNet_kwargs
        if scale_spec_patchNet_kwargs is not None:
            assert len(scale_spec_patchNet_kwargs) == nb_patch_nets
            for i in range(nb_patch_nets):
                self.ptch_kwargs[i]["output_shape"] = scale_spec_patchNet_kwargs[i]["patch_size"]
                # self.ptch_kwargs[i].update(scale_spec_patchNet_kwargs[i])
        self.patch_models = nn.ModuleList([
            PatchNet(**kwgs) for kwgs in self.ptch_kwargs
        ])

        # Build embedding heads:
        emb_slices = {}
        for i in range(nb_patch_nets):
            depth_patch_net = scale_spec_patchNet_kwargs[i].get("depth_level", 0)
            emb_slices[depth_patch_net] = [] if depth_patch_net not in emb_slices else emb_slices[depth_patch_net]
            nb_nets_at_depths = len(emb_slices[depth_patch_net])
            new_slc = (slice(None), slice(nb_nets_at_depths*self.output_fmaps, (nb_nets_at_depths+1)*self.output_fmaps))
            emb_slices[depth_patch_net].append((i, new_slc))

        self.emb_slices =  emb_slices

    def forward(self, input):
        encoded_states = []
        current = input
        for encode, downsample in zip(self.encoder_modules, self.downsampling_modules):
            current = encode(current)
            encoded_states.append(current)
            current = downsample(current)
        current = self.base_module(current)

        emb_outputs = []
        for encoded_state, upsample, skip, merge, decode, depth in reversed(list(zip(
                encoded_states, self.upsampling_modules, self.skip_modules, self.merge_modules, self.decoder_modules, range(len(self.decoder_modules))))):
            if depth < self.stop_decoder_at_depth:
                break
            current = upsample(current)
            current = merge(current, encoded_state)
            current = decode(current)

            # Attach possible emb. heads:
            if depth in self.emb_slices:
                for emb_slc in self.emb_slices[depth]:
                    emb_out = current[emb_slc[1]]
                    emb_outputs.append(emb_out)


        emb_outputs.reverse()

        return emb_outputs


    def construct_embedding_heads(self, depth):
        assert depth >= self.stop_decoder_at_depth
        return ConvNormActivation(self.pyramid_fmaps, self.output_fmaps, kernel_size=(1, 1, 1),
                                  dim=3,
                                  activation=None,
                                  normalization=None)

    def construct_merge_module(self, depth):
        if depth >= self.stop_decoder_at_depth:
            return MergePyramidNew(self.pyramid_fmaps, self.encoder_fmaps[depth])
        else:
            return None


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
                ResBlockAdvanced(f_out, f_inner=f_out, pre_kernel_size=(3, 3, 3),
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
        if depth >= self.stop_decoder_at_depth:
            # Let's try with a super-simple conv:
            return ConvNormActivation(self.pyramid_fmaps, self.pyramid_fmaps, kernel_size=(1, 3, 3),
                                            dim=3,
                                            activation="ReLU",
                                            num_groups_norm=16,
                                            normalization="GroupNorm")
        else:
            return None

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



class GeneralizedFeaturePyramidUNet3D(FeaturePyramidUNet3D):
    def __init__(self, depth, in_channels, encoder_fmaps, pyramid_fmaps,
                 res_blocks_3D,
                 scale_factor=2,
                 final_activation=None, # FIXME
                 upsampling_mode='nearest',
                 patchNet_kwargs=None,
                 output_fmaps=None,
                 decoder_crops=None,
                 stop_decoder_at_depth=0, # Sometimes we stop the decoder earlier
                 previous_output_from_stacked_models=False,
                 add_embedding_heads=False,
                 strided_res_blocks=False,
                 add_final_conv_in_res_block=False,
                 pre_kernel_size_res_block=(1,3,3),
                 legacy_resblock=None,
                 keep_raw=False,
                 **kwargs):
        # TODO: assert all this stuff
        self.strided_res_blocks = strided_res_blocks
        self.keep_raw = keep_raw
        self.add_final_conv_in_res_block = add_final_conv_in_res_block
        self.legacy_resblock = legacy_resblock
        self.depth = depth
        if isinstance(pre_kernel_size_res_block, list):
            pre_kernel_size_res_block = tuple(pre_kernel_size_res_block)
        self.pre_kernel_size_res_block = pre_kernel_size_res_block
        self.in_channels = in_channels
        self.pyramid_fmaps = pyramid_fmaps
        self.output_fmaps = pyramid_fmaps if output_fmaps is None else output_fmaps
        self.stop_decoder_at_depth = stop_decoder_at_depth
        self.res_blocks_3D = res_blocks_3D
        assert isinstance(previous_output_from_stacked_models, bool)
        self.previous_output_from_stacked_models = previous_output_from_stacked_models
        self.decoder_crops = decoder_crops if decoder_crops is not None else {}
        self.final_crop = self.decoder_crops.get(0, None)
        # assert len(self.decoder_crops) <= 1, "For the moment maximum one crop is supported"


        assert isinstance(encoder_fmaps, (list, tuple))
        self.encoder_fmaps = encoder_fmaps

        self.final_activation = [final_activation] if final_activation is not None else None


        # Parse scale factor
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

        # TODO: compute input size divisibiliy constraints

        super(FeaturePyramidUNet3D, self).__init__(depth)

        # Build patchNets:
        # patchNet_kwargs["latent_variable_size"] = pyramid_fmaps
        global_kwargs = patchNet_kwargs.pop("global", {})
        self.nb_patch_nets = nb_patch_nets = len(patchNet_kwargs.keys())
        self.ptch_kwargs = [deepcopy(global_kwargs) for _ in range(nb_patch_nets)]
        for i in range(nb_patch_nets):
            self.ptch_kwargs[i].update(patchNet_kwargs[i])
        self.patch_models = nn.ModuleList([
            PatchNet(**kwgs) for kwgs in self.ptch_kwargs
        ])

        # Build embedding heads:
        self.add_embedding_heads = add_embedding_heads
        if add_embedding_heads:
            emb_heads = {}
            for i in range(nb_patch_nets):
                depth_patch_net = patchNet_kwargs[i].get("depth_level", 0)
                emb_heads[depth_patch_net] = [] if depth_patch_net not in emb_heads else emb_heads[depth_patch_net]
                emb_heads[depth_patch_net].append(
                    self.construct_embedding_heads(depth_patch_net, nb_patch_net=i))

            self.emb_heads = nn.ModuleDict(
                {str(dpth): nn.ModuleList(emb_heads[dpth]) for dpth in emb_heads}
            )
        else:
            emb_slices = {}
            for i in range(nb_patch_nets):
                depth_patch_net = patchNet_kwargs[i].get("depth_level", 0)
                emb_slices[depth_patch_net] = [] if depth_patch_net not in emb_slices else emb_slices[depth_patch_net]
                emb_slices[depth_patch_net].append((i, self.construct_emb_slice(depth_patch_net, emb_slices[depth_patch_net])))

            self.emb_slices = emb_slices

    def construct_emb_slice(self, depth, previous_emb_slices):
        nb_nets_at_depths = len(previous_emb_slices)
        return (
        slice(None), slice(nb_nets_at_depths * self.output_fmaps, (nb_nets_at_depths + 1) * self.output_fmaps))

    def construct_embedding_heads(self, depth, nb_patch_net=None):
        assert depth >= self.stop_decoder_at_depth
        # TODO: generalize final activation!
        return ConvNormActivation(self.pyramid_fmaps, self.output_fmaps, kernel_size=(1, 1, 1),
                                  dim=3,
                                  activation='Sigmoid',
                                  normalization=None)


    def forward(self, *inputs):
        assert not self.keep_raw, "Not implemented yet at this level"
        # Modification for stacked architectures:
        # (previous output is inserted at depth 1)
        nb_inputs = len(inputs)
        assert nb_inputs <= 2
        input = inputs[0]
        previous_output = inputs[1] if nb_inputs == 2 else None

        encoded_states = []
        current = input
        for encode, downsample, depth in zip(self.encoder_modules, self.downsampling_modules,
                                      range(self.depth)):
            if depth == 1 and previous_output is not None:
                assert self.previous_output_from_stacked_models
                current = torch.cat((previous_output, current), dim=1)
            current = encode(current)
            encoded_states.append(current)
            current = downsample(current)
        current = self.base_module(current)

        emb_outputs = []
        for encoded_state, upsample, skip, merge, decode, depth in reversed(list(zip(
                encoded_states, self.upsampling_modules, self.skip_modules, self.merge_modules, self.decoder_modules, range(len(self.decoder_modules))))):
            if depth < self.stop_decoder_at_depth:
                break
            current = upsample(current)
            current = merge(current, encoded_state)
            current = decode(current)

            # Possible final crop:
            if depth == 0 and self.final_crop is not None:
                assert isinstance(self.final_crop, str)
                from inferno.io.volumetric.volumetric_utils import parse_data_slice
                crop_slice = (slice(None), slice(None)) + parse_data_slice(self.final_crop)
                current = current[crop_slice]

            if self.add_embedding_heads:
                if str(depth) in self.emb_heads:
                    for emb_head in self.emb_heads[str(depth)]:
                        emb_out = emb_head(current)
                        emb_outputs.append(emb_out)
            else:
                if depth in self.emb_slices:
                    for emb_slc in self.emb_slices[depth]:
                        emb_out = current[emb_slc[1]]
                        emb_outputs.append(emb_out)

        emb_outputs.reverse()

        return emb_outputs



    def construct_upsampling_module(self, depth):
        scale_factor = self.scale_factors[depth]
        if scale_factor[0] == 1:
            assert scale_factor[1] == scale_factor[2]

        # print(depth, scale_factor, self.decoder_crops.get(depth-1), )
        sampler = UpsampleAndCrop(scale_factor=scale_factor, mode=self.upsampling_mode,
                                  crop_slice=self.decoder_crops.get(depth-1))
        return sampler


    def construct_merge_module(self, depth):
        if depth >= self.stop_decoder_at_depth:
            return MergePyramidAndAutoCrop(self.pyramid_fmaps, self.encoder_fmaps[depth])
        else:
            return None

    def concatenate_res_blocks(self, f_in, f_out, blocks_spec, depth):
        """
        Concatenate multiple residual blocks according to the config file
        """
        # TODO: generalize
        assert f_out % 16 == 0, "Not divisible for group norm!"

        blocks_list = []
        if not self.strided_res_blocks:
            # if depth == 0:
            #     # assert all([not is_3D for is_3D in blocks_spec]), "All blocks at highest level should be 2D"
            #     # Add by the default the first block:
            #     blocks_list.append(ResBlockAdvanced(f_in, f_inner=f_out, pre_kernel_size=(1, 3, 3),
            #                          inner_kernel_size=(3, 3, 3),
            #                          activation="ReLU",
            #                          normalization="GroupNorm",
            #                          num_groups_norm=16,
            #                          dilation=(1,2,2)
            #                          ))
            #     blocks_spec.pop(0)
            #     f_in = f_out

            # Concatenate possible additional blocks:
            for is_3D in blocks_spec:
                assert isinstance(is_3D, bool)
                if is_3D:
                    blocks_list.append(ResBlockAdvanced(f_in, f_inner=f_out, pre_kernel_size=self.pre_kernel_size_res_block,
                                     inner_kernel_size=(3, 3, 3),
                                     activation="ReLU",
                                     normalization="GroupNorm",
                                     num_groups_norm=16,
                                                        add_final_conv=self.add_final_conv_in_res_block,
                                                        legacy_version=self.legacy_resblock
                                     ))
                else:
                    blocks_list.append(ResBlockAdvanced(f_in, f_inner=f_out, pre_kernel_size=(1, 3, 3),
                                     inner_kernel_size=(1, 3, 3),
                                     activation="ReLU",
                                     normalization="GroupNorm",
                                     num_groups_norm=16,
                                     ))
                f_in = f_out
        else:
            dilation = (1, 3, 3) if depth <= 1 else (2, 2, 2)
            if depth == 0:
                assert all([not is_3D for is_3D in blocks_spec]), "All blocks at highest level should be 2D"
                # Add by the default the first block:
                blocks_list.append(ResBlockAdvanced(f_in, f_inner=f_out, pre_kernel_size=(3, 3, 3),
                                                    inner_kernel_size=(1, 3, 3),
                                                    activation="ReLU",
                                                    normalization="GroupNorm",
                                                    num_groups_norm=16,
                                                    dilation=dilation
                                                    ))
                blocks_spec.pop(0)
                f_in = f_out

            # Concatenate possible additional blocks:
            for is_3D in blocks_spec:
                assert isinstance(is_3D, bool)
                if is_3D:
                    blocks_list.append(ResBlockAdvanced(f_in, f_inner=f_out, pre_kernel_size=(1, 3, 3),
                                                        inner_kernel_size=(3, 3, 3),
                                                        activation="ReLU",
                                                        normalization="GroupNorm",
                                                        num_groups_norm=16,
                                                        dilation=dilation
                                                        ))
                else:
                    blocks_list.append(ResBlockAdvanced(f_in, f_inner=f_out, pre_kernel_size=(1, 3, 3),
                                                        inner_kernel_size=(1, 3, 3),
                                                        activation="ReLU",
                                                        normalization="GroupNorm",
                                                        num_groups_norm=16,
                                                        dilation=dilation
                                                        ))
                f_in = f_out

        return nn.Sequential(*blocks_list)


    def construct_encoder_module(self, depth):
        f_in = self.encoder_fmaps[depth - 1] if depth != 0 else self.in_channels
        f_out = self.encoder_fmaps[depth]

        # Increase input channels if we expect outputs from previous stacked models:
        if depth == 1 and self.previous_output_from_stacked_models:
            f_in = self.pyramid_fmaps + self.encoder_fmaps[0]
            assert f_in <= f_out, "Bottleneck! Output channels are less the input ones"

        # Build blocks:
        blocks_spec = deepcopy(self.res_blocks_3D[depth])

        if depth == 0:
            first_conv = ConvNormActivation(f_in, f_out, kernel_size=(1, 5, 5),
                                           dim=3,
                                           activation="ReLU",
                                           num_groups_norm=16,
                                           normalization="GroupNorm")
            # Here the block has a different number of inpiut channels:
            res_block = self.concatenate_res_blocks(f_out, f_out, blocks_spec, depth)
            res_block = nn.Sequential(first_conv, res_block)
        else:
            res_block = self.concatenate_res_blocks(f_in, f_out, blocks_spec, depth)

        return res_block

    def construct_decoder_module(self, depth):
        if depth >= self.stop_decoder_at_depth:
            # Let's try with a super-simple conv:
            return ConvNormActivation(self.pyramid_fmaps, self.pyramid_fmaps, kernel_size=(1, 3, 3),
                                            dim=3,
                                            activation="ReLU",
                                            num_groups_norm=16,
                                            normalization="GroupNorm")
        else:
            return None

    def construct_base_module(self):
        f_in = self.encoder_fmaps[self.depth - 1]
        # f_intermediate = self.encoder_fmaps[self.depth]
        raise Warning("Last 512 channels not used")
        f_out = self.pyramid_fmaps
        blocks_spec = deepcopy(self.res_blocks_3D[self.depth])
        return self.concatenate_res_blocks(f_in, f_out, blocks_spec, self.depth)


class GeneralizedUNet3D(GeneralizedFeaturePyramidUNet3D):
    def __init__(self, *super_args,
                 decoder_fmaps, res_blocks_decoder_3D,
                 **super_kwargs):
        assert isinstance(decoder_fmaps, (list, tuple))
        self.decoder_fmaps = decoder_fmaps
        self.res_blocks_decoder_3D = res_blocks_decoder_3D
        super(GeneralizedUNet3D, self).__init__(*super_args, **super_kwargs)

    def construct_merge_module(self, depth):
        if depth >= self.stop_decoder_at_depth:
            return MergePyramidAndAutoCrop(self.decoder_fmaps[depth], self.encoder_fmaps[depth])
        else:
            return None

    def construct_upsampling_module(self, depth):
        # First we need to reduce the numer of channels:
        conv = ConvNormActivation(self.decoder_fmaps[depth+1], self.decoder_fmaps[depth], kernel_size=(1, 1, 1),
                           dim=3,
                           activation="ReLU",
                           num_groups_norm=16,
                           normalization="GroupNorm")

        scale_factor = self.scale_factors[depth]
        if scale_factor[0] == 1:
            assert scale_factor[1] == scale_factor[2]

        # print(scale_factor, depth, self.decoder_crops.get(depth+1, None))
        sampler = UpsampleAndCrop(scale_factor=scale_factor, mode=self.upsampling_mode,
                                  crop_slice=self.decoder_crops.get(depth+1, None))

        return nn.Sequential(conv, sampler)

    def construct_decoder_module(self, depth):
        # FIXME: this is broken. Assume to output only one embedding at the highest scale!
        assert self.stop_decoder_at_depth == 0
        if depth >= self.stop_decoder_at_depth:
            f_in = self.decoder_fmaps[depth]
            f_out = self.decoder_fmaps[depth]

            # Build blocks:
            blocks_spec = deepcopy(self.res_blocks_decoder_3D[depth])
            # Remark: embeddings are also with ReLU and batchnorm!
            res_block = self.concatenate_res_blocks(f_in, f_out, blocks_spec, depth)
            if depth == 0:
                last_conv = ConvNormActivation(f_out, f_out, kernel_size=(1, 5, 5),
                           dim=3,
                           activation="ReLU",
                           num_groups_norm=16,
                           normalization="GroupNorm")
                res_block = nn.Sequential(res_block, last_conv)
            return res_block
            # Let's try with a super-simple conv:
            # if depth != 0:
            #     return ConvNormActivation(f_in, f_out, kernel_size=(1, 3, 3),
            #                           dim=3,
            #                           activation="ReLU",
            #                           num_groups_norm=16,
            #                           normalization="GroupNorm")
            # else:
            #     return ConvNormActivation(f_in, f_out, kernel_size=(1, 3, 3),
            #                           dim=3,
            #                           activation=None,
            #                           normalization=None)
        else:
            return None

    def construct_base_module(self):
        f_in = self.encoder_fmaps[self.depth - 1]
        # f_intermediate = self.encoder_fmaps[self.depth]
        f_out = self.decoder_fmaps[self.depth]
        blocks_spec = deepcopy(self.res_blocks_3D[self.depth])
        return self.concatenate_res_blocks(f_in, f_out, blocks_spec, self.depth)

    def construct_emb_slice(self, depth, previous_emb_slices):
        return (slice(None), slice(None))

    def construct_embedding_heads(self, depth):
        assert depth >= self.stop_decoder_at_depth
        # TODO: generalize final activation!
        return ConvNormActivation(self.decoder_fmaps[depth], self.output_fmaps, kernel_size=(1, 1, 1),
                                  dim=3,
                                  activation='Sigmoid',
                                  normalization=None)

    def construct_downsampling_module(self, depth):
        scale_factor = self.scale_factors[depth]
        # kernel = (1,3,3)
        # assert all(k>=sc for k, sc in zip(kernel, scale_factor))
        sampler = ConvNormActivation(self.encoder_fmaps[depth], self.encoder_fmaps[depth],
                           kernel_size=scale_factor,
                           dim=3,
                           stride=scale_factor,
                           valid_conv=True,
                           activation="ReLU",
                           num_groups_norm=16,
                           normalization="GroupNorm")
        # sampler = nn.MaxPool3d(kernel_size=scale_factor,
        #                        stride=scale_factor,
        #                        padding=0)
        return sampler

class MultiScaleInputsUNet3D(GeneralizedUNet3D):
    def __init__(self,
                 *super_args,
                 add_foreground_prediction_module=False,
                 **super_kwargs):
        """
        The crops in the decode have been moved after ASPP, so that we leave the big context available
        """
        super(MultiScaleInputsUNet3D, self).__init__(*super_args, **super_kwargs)

        # Construct the extra modules:
        self.autopad_first_encoding = AutoPad()


        self.add_foreground_prediction_module = add_foreground_prediction_module
        self.foreground_module = None
        if add_foreground_prediction_module:
            # self.foreground_module = nn.Sequential(
            #     ConvNormActivation(in_channels=self.models[-1].output_fmaps, out_channels=int(self.models[-1].output_fmaps/4),
            #                kernel_size=(1, 3, 3), dim=3, activation='ReLU', normalization="GroupNorm", num_groups_norm=16),
            #     ConvNormActivation(in_channels=int(self.models[-1].output_fmaps/4), out_channels=1,
            #                        kernel_size=(1, 3, 3), dim=3, activation='Sigmoid', normalization=None))

            self.foreground_module = ConvNormActivation(self.decoder_fmaps[0],
                                          out_channels=1,
                                          kernel_size=1,
                                          dim=3,
                                          activation='Sigmoid',
                                          normalization=None,
                                          num_groups_norm=16)




    def forward(self, *inputs):
        nb_inputs = len(inputs)
        assert nb_inputs == 2, "Only two inputs accepted for the moment"

        encoded_states = []
        current = inputs[0]
        for encode, downsample, depth in zip(self.encoder_modules, self.downsampling_modules,
                                      range(self.depth)):
            from speedrun.log_anywhere import log_image, log_embedding, log_scalar
            if depth == 1:
                current_lvl0_padded = self.autopad_first_encoding(current, inputs[1].shape)
                # -------- DEBUG -----------
                # TODO: pad input and check if it fits...
                inputs_DS = inputs[0][:,:,:,::2,::2]
                inputs_DS_padded = self.autopad_first_encoding(inputs_DS, inputs[1].shape)
                log_image("input_ds", inputs_DS_padded)
                log_image("mid_layer", current_lvl0_padded)
                # -------- DEBUG -----------
                current = torch.cat((current_lvl0_padded, inputs[1]), dim=1)
                current = encode(current)
            else:
                current = encode(current)
            encoded_states.append(current)
            if depth > 0:
                log_image("encoder_layer_depth_{}".format(depth), current)
            current = downsample(current)
        current = self.base_module(current)
        log_image("encoder_layer_depth_base", current)

        emb_outputs = []
        for encoded_state, upsample, skip, merge, decode, depth in reversed(list(zip(
                encoded_states, self.upsampling_modules, self.skip_modules, self.merge_modules, self.decoder_modules, range(len(self.decoder_modules))))):
            if depth < self.stop_decoder_at_depth:
                break
            current = upsample(current)
            current = merge(current, encoded_state)
            current = decode(current)

            # FIXME: use reversed to make sure that they are outputed in the right order!
            # Legacy problem, because previous networks are broken...
            if self.add_embedding_heads:
                if str(depth) in self.emb_heads:
                    # for emb_head in reversed(self.emb_heads[str(depth)]):
                    for emb_head in self.emb_heads[str(depth)]:
                        emb_out = emb_head(current)
                        emb_outputs.append(emb_out)
            else:
                if depth in self.emb_slices:
                    # for emb_slc in reversed(self.emb_slices[depth]):
                    for emb_slc in self.emb_slices[depth]:
                        emb_out = current[emb_slc[1]]
                        emb_outputs.append(emb_out)


        emb_outputs.reverse()

        if self.foreground_module is not None:
            emb_outputs.append(self.foreground_module(current))

        if self.keep_raw:
            emb_outputs = emb_outputs + list(inputs)

        return emb_outputs

    def construct_merge_module(self, depth):
        if depth >= self.stop_decoder_at_depth:
            return MergePyramidAndAutoCrop(self.decoder_fmaps[depth], self.encoder_fmaps[depth])
        else:
            return None


    def construct_embedding_heads(self, depth, nb_patch_net=None):
        assert nb_patch_net is not None
        ptch_kwargs = self.ptch_kwargs[nb_patch_net]
        ASPP_kwargs = deepcopy(ptch_kwargs.get("ASPP_kwargs", {}))
        if ASPP_kwargs.pop("use_ASPP", True):
            dilations = ASPP_kwargs.pop("dilations", [[1,6,6], [1,12,12], [3,1,1]])
            assert isinstance(dilations, list) and all(isinstance(dil, list) for dil in dilations)
            dilations = [tuple(dil) for dil in dilations]
            from vaeAffs.models.ASPP import ASPP3D
            ASPP_inner_planes = ASPP_kwargs.pop("inner_planes", self.decoder_fmaps[depth])
            emb_head = ASPP3D(inplanes=self.decoder_fmaps[depth],
                                 inner_planes=ASPP_inner_planes,
                                 output_planes=ptch_kwargs["latent_variable_size"],
                                 dilations=dilations,
                                 num_norm_groups=16,
                                 **ASPP_kwargs)
        else:
            norm = "GroupNorm" if ASPP_kwargs.get("apply_final_norm", True) else None
            emb_head = ConvNormActivation(self.decoder_fmaps[depth],
                                             out_channels=ptch_kwargs["latent_variable_size"],
                                             kernel_size=1,
                                             dim=3,
                                             activation=ASPP_kwargs.get("final_act", "ReLU"),
                                             normalization=norm,
                                             num_groups_norm=16)

        crop = self.decoder_crops.get(depth, None)
        emb_head = nn.Sequential(emb_head, Crop(crop)) if crop is not None else emb_head

        return emb_head

    def construct_encoder_module(self, depth):
        if depth == 0:
            f_in = self.in_channels
        elif depth == 1:
            f_in = self.encoder_fmaps[depth - 1] + self.in_channels
        else:
            f_in = self.encoder_fmaps[depth - 1]
        f_out = self.encoder_fmaps[depth]

        # Increase input channels if we expect outputs from previous stacked models:
        if depth == 1 and self.previous_output_from_stacked_models:
            f_in = self.pyramid_fmaps + self.encoder_fmaps[0]
            assert f_in <= f_out, "Bottleneck! Output channels are less the input ones"

        # Build blocks:
        blocks_spec = deepcopy(self.res_blocks_3D[depth])

        if depth == 0:
            first_conv = ConvNormActivation(f_in, f_out, kernel_size=(1, 5, 5),
                                           dim=3,
                                           activation="ReLU",
                                           num_groups_norm=16,
                                           normalization="GroupNorm")
            # Here the block has a different number of inpiut channels:
            res_block = self.concatenate_res_blocks(f_out, f_out, blocks_spec, depth)
            res_block = nn.Sequential(first_conv, res_block)
        else:
            res_block = self.concatenate_res_blocks(f_in, f_out, blocks_spec, depth)

        return res_block


class GeneralizedStackedPyramidUNet3D(nn.Module):
    def __init__(self,
                 nb_stacked,
                 models_kwargs,
                 models_to_train,
                 stacked_upscl_fact=None,
                 type_of_model="GeneralizedFeaturePyramidUNet3D",
                 detach_stacked_models=True,
                 nb_inputs_per_model=1
                 ):
        super(GeneralizedStackedPyramidUNet3D, self).__init__()
        assert isinstance(nb_stacked, int)
        self.nb_stacked = nb_stacked
        self.nb_inputs_per_model = nb_inputs_per_model

        # Collect models kwargs:
        models_kwargs = deepcopy(models_kwargs)
        global_kwargs = models_kwargs.pop("global", {})
        self.models_kwargs = [deepcopy(global_kwargs) for _ in range(nb_stacked)]
        for mdl in range(nb_stacked):
            if mdl in models_kwargs:
                self.models_kwargs[mdl].update(models_kwargs[mdl])
            # # All models should expect a previous input, apart from the first one:
            # if mdl > 0:
            #     self.models_kwargs[mdl]["previous_output_from_stacked_models"] = True

        # Build models (now PatchNets are also automatically built here):
        if type_of_model == "GeneralizedFeaturePyramidUNet3D":
            model_class = GeneralizedFeaturePyramidUNet3D
        elif type_of_model == "GeneralizedUNet3D":
            model_class = GeneralizedUNet3D
        elif type_of_model == "MultiScaleInputsUNet3D":
            model_class = MultiScaleInputsUNet3D
        else:
            raise ValueError
        self.type_of_model = type_of_model
        self.detach_stacked_models = detach_stacked_models
        self.models = nn.ModuleList([
            model_class(**kwargs) for kwargs in self.models_kwargs
        ])

        # Collect patchNet kwargs:
        assert isinstance(models_to_train, list)
        assert all(mdl < nb_stacked for mdl in models_to_train)
        self.models_to_train = models_to_train
        self.last_model_to_train = np.array(models_to_train).max()
        collected_patchNet_kwargs = []
        trained_patchNets = []
        j = 0
        for mdl in range(nb_stacked):
            all_patchNet_kwargs = self.models[mdl].ptch_kwargs
            for nb_ptchNet, patchNet_kwargs in enumerate(all_patchNet_kwargs):
                if mdl in models_to_train:
                    trained_patchNets.append(j)
                current_kwargs = deepcopy(patchNet_kwargs)
                current_kwargs["model_number"] = mdl
                current_kwargs["patchNet_number"] = nb_ptchNet
                collected_patchNet_kwargs.append(current_kwargs)
                j += 1
        self.collected_patchNet_kwargs = collected_patchNet_kwargs
        self.trained_patchNets = trained_patchNets

        # Freeze-parameters for models that are not trained:
        for mdl in range(nb_stacked):
            if mdl not in self.models_to_train:
                for param in self.models[mdl].parameters():
                    param.requires_grad = False

        # # Build crop-modules:
        # from vaeAffs.transforms import DownsampleAndCrop3D
        # self.downscale_and_crop = downscale_and_crop if downscale_and_crop is not None else {}
        # # TODO: deprecated (now an auto-crop is used)
        # self.crop_transforms = {mdl: DownsampleAndCrop3D(**self.downscale_and_crop[mdl]) for mdl in self.downscale_and_crop}

        self.stacked_upscl_fact = stacked_upscl_fact if stacked_upscl_fact is not None else []
        assert len(self.stacked_upscl_fact) == self.nb_stacked - 1
        self.upsample_modules = nn.ModuleList([
            Upsample(scale_factor=tuple(scl_fact), mode="nearest") for scl_fact in self.stacked_upscl_fact
        ])


        # TODO: not sure it changes anything...
        self.fix_batchnorm_problem()
        self.first_debug_print = True

    def fix_batchnorm_problem(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, *inputs):
        assert len(inputs) == self.nb_stacked * self.nb_inputs_per_model
        if self.nb_inputs_per_model != 1:
            inputs = tuple(inputs[i*self.nb_inputs_per_model:(i+1)*self.nb_inputs_per_model]  for i in range(self.nb_stacked))

        # Apply stacked models:
        last_output = None
        output_features = []
        for mdl in range(self.nb_stacked):
            import contextlib
            # Run in inference mode if we do not need to train patches of this model:
            no_grad = torch.no_grad if mdl not in self.models_to_train else contextlib.nullcontext
            with no_grad():
                if mdl == 0:
                    if self.nb_inputs_per_model == 1:
                        current_outputs = self.models[mdl](inputs[mdl])
                    else:
                        current_outputs = self.models[mdl](*inputs[mdl])
                else:
                    # Pass previous output:
                    # choice = np.random.randint(8)
                    # if choice == 0:
                    #     current_outputs = self.models[mdl](torch.zeros_like(inputs[mdl]), last_output)
                    # elif choice == 1:
                    #     current_outputs = self.models[mdl](inputs[mdl], torch.zeros_like(last_output))
                    # else:
                    # current_outputs = self.models[mdl](inputs[mdl], last_output)
                    assert self.nb_inputs_per_model == 1
                    current_outputs = self.models[mdl](torch.cat((inputs[mdl], last_output), dim=1))

            # print("Mdl {}, memory {} - {}".format(mdl, torch.cuda.memory_allocated(0)/1000000, torch.cuda.memory_cached(0)/1000000,))
            # torch.cuda.empty_cache()
            if mdl in self.models_to_train:
                output_features += current_outputs

            # Check if we should stop because next models are not trained:
            if mdl+1 > self.last_model_to_train:
                # if self.first_debug_print and last_output.device == torch.device('cuda:0'):
                #     print("Output shape: ", current_outputs[0].shape, " || Input shape: ", inputs[mdl].shape)
                break

            # Get the first output and prepare it to be inputed to the next model (if not the last):
            # (avoid backprop between models atm)
            if mdl != self.nb_stacked - 1:
                last_output = current_outputs[0]
                if self.detach_stacked_models:
                    last_output = last_output.detach()

                # Get input shape of the next stacked model:
                crp_shp = inputs[mdl+1].shape
                # Get next-input shape before to upscale:
                crp_shp = crp_shp[:2] + tuple(int(crp_shp[i+2]/self.stacked_upscl_fact[mdl][i]) for i in range(3))
                # Auto-crop:
                # if self.first_debug_print and last_output.device == torch.device('cuda:0'):
                #     print("Output shape: ", last_output.shape, " || Target shape: ", crp_shp, " || Input shape: ", inputs[mdl].shape)
                last_output = auto_crop_tensor_to_shape(last_output, crp_shp)
                # Up-sample:
                last_output = self.upsample_modules[mdl](last_output)
                from speedrun.log_anywhere import log_image, log_embedding, log_scalar
                # print(last_output.device)
                # if last_output.device == torch.device('cuda:0'):
                #     log_image("output_mdl_{}".format(mdl), last_output)

        self.first_debug_print = False
        return output_features


class IntersectOverUnionUNet(GeneralizedStackedPyramidUNet3D):
    def __init__(self, *super_args, **super_kwargs):
        super(IntersectOverUnionUNet, self).__init__(*super_args, **super_kwargs)

        # Freeze-parameters of the pretrained model:
        for mdl in range(self.nb_stacked):
            for param in self.models[mdl].parameters():
                param.requires_grad = False

        # Build extra module:
        self.embedding_dimension = self.models[0].output_fmaps
        self.IoU_module = IntersectOverUnionModule(embedding_dimension=self.embedding_dimension)

    def forward(self, *inputs):
        with torch.no_grad():
            outputs = super(IntersectOverUnionUNet, self).forward(*inputs)
        return outputs

class IntersectOverUnionModule(nn.Module):
    def __init__(self, embedding_dimension=128):
        super(IntersectOverUnionModule, self).__init__()
        self.embedding_dimension = embedding_dimension
        # Build some 1D convolutions:
        # TODO: find a better solution and avoid padding with kernel 3...
        self.block1 = nn.Sequential(
            self.build_normalized_conv(embedding_dimension, embedding_dimension, (3, )),
             self.build_normalized_conv(embedding_dimension, embedding_dimension, (3, )))
        self.max_pool = nn.MaxPool1d(kernel_size=(2,), stride=(2,))
        self.block2 = nn.Sequential(
             self.build_normalized_conv(embedding_dimension, int(embedding_dimension/2), (1,)),
             self.build_normalized_conv(int(embedding_dimension/2), int(embedding_dimension/4), (1,)),
             self.build_normalized_conv(int(embedding_dimension/4), int(embedding_dimension/8), (1,)),
        )
        self.final_conv = ConvNormActivation(in_channels=int(embedding_dimension/8), out_channels=1,
                           kernel_size=(1,), dim=1, activation="Sigmoid")
        self.final_conv_2 = ConvNormActivation(in_channels=int(embedding_dimension), out_channels=1,
                                             kernel_size=(1,), dim=1, activation="Sigmoid")


    def build_normalized_conv(self, in_ch, out_ch, kernel_size):
        return ConvNormActivation(in_channels=in_ch, out_channels=out_ch,
                           kernel_size=kernel_size, dim=1, activation="ReLU",
                                            num_groups_norm=16,
                                            normalization="GroupNorm")

    def forward(self, embeddings_1, embeddings_2):
        # current = torch.cat((embeddings_1, embeddings_2), dim=2)
        # current = self.block1(current)
        # current = self.max_pool(current)
        # current = self.block2(current)
        # return self.final_conv(current)

        # current = torch.cat((embeddings_1, embeddings_2), dim=1)
        return self.final_conv_2(embeddings_1)



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
                               # stride=scale_factor,
                               # ceil_mode=True,
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


class StackedAffinityNet(nn.Module):
    def __init__(self, path_model, nb_offsets, *super_args, **super_kwargs):
        super(StackedAffinityNet, self).__init__()

        self.stacked_model = torch.load(path_model)["_model"]

        nb_pyr_maps = self.stacked_model.pyr_models[0].pyramid_fmaps

        self.final_module = nn.Sequential(
            Conv3D(in_channels=nb_pyr_maps * 3, out_channels=nb_offsets,
                   kernel_size=(1, 5, 5))
            # ResBlockAdvanced(f_in=nb_pyr_maps * 3, f_inner=nb_pyr_maps, f_out=nb_offsets, pre_kernel_size=(1, 3, 3),
            #                  inner_kernel_size=(1, 3, 3), num_groups_norm=16, dilation=4,
            #                  apply_final_activation=False,
            #                  apply_final_normalization=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, *raw):
        # Hack because this model was trained with affs:
        mdl = self.stacked_model

        with torch.no_grad():
            feat_pyr = mdl(*raw)

        # FIXME:
        upscaled_feat_pyr = [feat_pyr[0],
                             feat_pyr[3],
                             feat_pyr[6],]

        upscaled_feat_pyr[0] = mdl.crop_and_upsample(mdl.crop_and_upsample(upscaled_feat_pyr[0], 0), 1)
        upscaled_feat_pyr[1] = mdl.crop_and_upsample(upscaled_feat_pyr[1], 1)

        output = self.final_module(torch.cat(tuple(upscaled_feat_pyr[:3]), dim=1))
        return self.sigmoid(output)


class GeneralizedAffinitiesFromEmb(nn.Module):
    def __init__(self, path_model, nb_offsets, train_backbone=False, use_ASPP_module=False, reload_backbone=True,
                 ASPP_inner_planes=64, ASPP_dilations=None, prediction_index=None):
        super(GeneralizedAffinitiesFromEmb, self).__init__()

        self.path_model = path_model
        self.train_backbone = train_backbone
        self.reload_backbone = reload_backbone
        self.load_backbone()

        output_maps = self.stacked_model.models[-1].output_fmaps

        if use_ASPP_module:
            from vaeAffs.models.ASPP import ASPP3D
            if ASPP_dilations is None:
                ASPP_dilations = [(1,18,18), (1,36,36), (1,54,54), (3, 18, 18)]
            else:
                assert isinstance(ASPP_dilations, list) and all(isinstance(dil, list) for dil in ASPP_dilations)
                ASPP_dilations = [tuple(dil) for dil in ASPP_dilations]
            aspp_module = ASPP3D(inplanes=output_maps, inner_planes=ASPP_inner_planes, dilations=ASPP_dilations, num_norm_groups=16)
            raise NotImplementedError("Replace Conv3D with new Conv")
            final_conv = Conv3D(in_channels=ASPP_inner_planes, out_channels=nb_offsets,
                       kernel_size=1)
            self.final_module = nn.Sequential(aspp_module, final_conv)
        else:
            raise NotImplementedError("Replace Conv3D with new Conv")
            self.final_module = nn.Sequential(
                Conv3D(in_channels=output_maps, out_channels=nb_offsets,
                       kernel_size=(1, 5, 5))
            )

        self.sigmoid = nn.Sigmoid()
        self.prediction_index = prediction_index

    def forward(self, *inputs):
        with torch.no_grad():
            current = self.stacked_model(*inputs)
        if self.prediction_index is not None:
            current = current[self.prediction_index]
        else:
            current = current[-1]
        from speedrun.log_anywhere import log_image, log_embedding, log_scalar
        if current.device == torch.device(0):
            log_image("embeddings", current)
        current = self.final_module(current)
        return self.sigmoid(current)

    def load_state_dict(self, state_dict):
        super(GeneralizedAffinitiesFromEmb, self).load_state_dict(state_dict)
        # Reload backbone:
        if self.reload_backbone:
            self.load_backbone()

    def load_backbone(self):
        self.stacked_model = torch.load(self.path_model)["_model"]

        # Freeze-parameters for models that are not trained:
        if not self.train_backbone:
            print("Backbone parameters freezed!")
            for param in self.stacked_model.parameters():
                param.requires_grad = False



class SmartAffinitiesFromEmb(nn.Module):
    def __init__(self, path_model, prediction_indices, train_backbone=False, reload_backbone=True,
                 ASPP_kwargs=None, layers_kwargs=None):
        super(SmartAffinitiesFromEmb, self).__init__()

        self.path_model = path_model
        self.train_backbone = train_backbone
        self.reload_backbone = reload_backbone
        self.load_backbone()


        self.prediction_indices = prediction_indices
        output_maps = self.backbone_model.models[-1].output_fmaps * len(prediction_indices) * 2 + 3

        # from vaeAffs.models.ASPP import ASPP3D
        # aspp_module = ASPP3D(inplanes=output_maps, num_norm_groups=16, output_planes=1, **ASPP_kwargs)

        # Build some 1x1 layers:
        layers = [ConvNormActivation(in_channels=output_maps, **layers_kwargs)]
        for _ in range(4):
            layers.append(ConvNormActivation(in_channels=layers_kwargs["out_channels"], **layers_kwargs))

        layers.append(ConvNormActivation(in_channels=layers_kwargs.get("out_channels"),
                                         out_channels=1,
                                         kernel_size=(1,1,1),
                                         dim=3,
                                         activation=None,
                                         normalization=None))
        self.final_module = nn.Sequential(*layers)

        self.sigmoid = nn.Sigmoid()

    def forward(self, *inputs):
        inputs_backbone = inputs[:-1]

        with torch.no_grad():
            current = self.backbone_model(*inputs_backbone)
        current = [current[idx] for idx in self.prediction_indices]
        current = torch.cat(current, dim=1)
        inverse_offset = tuple(-int(inputs[-1][0,i,0,0,0].item()) for i in range(3))
        rolled_current = torch.roll(current, inverse_offset, dims=(2,3,4))
        from speedrun.log_anywhere import log_image, log_embedding, log_scalar
        log_image("concatenated_embs_0", current)
        log_image("concatenated_embs_1", rolled_current)

        current = torch.cat([auto_crop_tensor_to_shape(inputs[-1], current.shape),
                             current,
                             rolled_current], dim=1)

        # from speedrun.log_anywhere import log_image, log_embedding, log_scalar
        # if current.device == torch.device(0):
        #     log_image("embeddings", current)

        # Concatenate all channels:
        current = self.final_module(current)
        return self.sigmoid(current)

    def load_state_dict(self, state_dict):
        super(SmartAffinitiesFromEmb, self).load_state_dict(state_dict)
        # Reload backbone:
        if self.reload_backbone:
            self.load_backbone()

    def load_backbone(self):
        self.backbone_model = torch.load(self.path_model)["_model"]

        # Freeze-parameters for models that are not trained:
        if not self.train_backbone:
            print("Backbone parameters freezed!")
            for param in self.backbone_model.parameters():
                param.requires_grad = False




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
