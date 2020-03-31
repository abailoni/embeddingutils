from inferno.extensions.containers.graph import Graph, Identity

from inferno.extensions.layers.convolutional import ConvELU3D, Conv3D, BNReLUConv3D
from inferno.extensions.layers.convolutional import ConvELU2D, Conv2D, BNReLUConv2D
from inferno.extensions.layers.reshape import Concatenate, Sum

from embeddingutils.models.submodules import SuperhumanSNEMIBlock, ConvGRU, ShakeShakeMerge, Upsample
from embeddingutils.models.submodules import ResBlockAdvanced

import torch
import torch.nn as nn
from copy import deepcopy
from inferno.extensions.layers.convolutional import ConvNormActivation

import numpy as np





class Crop(nn.Module):
    """
    Crop a tensor according to the given string representing the crop slice
    """
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


class UpsampleAndCrop(nn.Module):
    """
    Combination of Upsample and Crop
    """
    def __init__(self, scale_factor, mode,
                                  crop_slice=None):
        super(UpsampleAndCrop, self).__init__()
        self.upsampler = Upsample(scale_factor=scale_factor, mode=mode)
        self.crop_module = Crop(crop_slice=crop_slice)

    def forward(self, input):
        if isinstance(input, tuple):
            input = input[0]
        input = self.crop_module(input)
        output = self.upsampler(input)
        return output


class MergePyramidAndAutoCrop(nn.Module):
    """
    Used in the UNet decoder to merge skip connections from feature maps at lower scales
    """
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
    """
    Used to auto-pad the multiple UNet inputs passed at different resolutions
    """
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

        # Legacy:
        pre_kernel_size = (3, 3, 3)
        num_groups_norm = 1

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
                                               legacy_version=None,
                                               apply_final_activation=False,
                                               apply_final_normalization=False)
        self.final_activation = nn.Sigmoid()

    def forward(self, encoded_variable):
        x = self.linear_base(encoded_variable)
        N = x.shape[0]
        reshaped = x.view(N, -1, *self.min_path_shape)

        out = self.decoder_module(reshaped)
        out = self.final_activation(out)
        return out


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



class MultiScaleInputsUNet3D(EncoderDecoderSkeleton):
    def __init__(self, depth,
                 in_channels,
                 encoder_fmaps,
                 add_foreground_prediction_module=False,
                 foreground_prediction_kwargs=None,
                 number_multiscale_inputs=1,
                 decoder_fmaps=None,
                 scale_factor=2,
                 res_blocks_specs=None,
                 res_blocks_specs_decoder=None,
                 upsampling_mode='nearest',
                 patchNet_kwargs=None,
                 decoder_crops=None,
                 return_input=False):
        """
        :param res_blocks_specs: None or list of booleans (lenght depth+1) specifying how many resBlocks we should concatenate
        at each level. Example:
            [
                [False, False], # Two 2D ResBlocks at the highest level
                [True],         # One 3D ResBlock at the second level of the UNet hierarchy
                [True, True],   # Two 3D ResBlock at the third level of the UNet hierarchy
            ]
        :param upsampling_mode:
        :param patchNet_kwargs:
        :param decoder_crops:
        :param decoder_fmaps: By default equal to the encoder_fmaps
        :param return_input:
        :param kwargs:
        """
        assert isinstance(return_input, bool)
        self.return_input = return_input

        assert isinstance(depth, int)
        self.depth = depth

        assert isinstance(in_channels, int)
        self.in_channels = in_channels

        assert isinstance(upsampling_mode, str)
        self.upsampling_mode = upsampling_mode

        assert isinstance(number_multiscale_inputs, int)
        self.number_multiscale_inputs = number_multiscale_inputs

        def assert_depth_args(f_maps):
            assert isinstance(f_maps, (list, tuple))
            assert len(f_maps) == depth + 1

        # Parse feature maps:
        assert_depth_args(encoder_fmaps)
        self.encoder_fmaps = encoder_fmaps
        if decoder_fmaps is None:
            # By default use symmetric architecture:
            self.decoder_fmaps = encoder_fmaps
        else:
            assert_depth_args(decoder_fmaps)
            assert decoder_fmaps[-1] == encoder_fmaps[-1], "Number of layers at the base module should be the same"
            self.decoder_fmaps = decoder_fmaps

        # Parse scale factor
        if isinstance(scale_factor, int):
            scale_factor = [scale_factor, ] * depth
        scale_factors = scale_factor
        normalized_factors = []
        for scale_factor in scale_factors:
            assert isinstance(scale_factor, (int, list, tuple))
            if isinstance(scale_factor, int):
                scale_factor = self.dim * [scale_factor]
            assert len(scale_factor) == self.dim
            normalized_factors.append(scale_factor)
        self.scale_factors = normalized_factors

        # Parse res-block specifications:
        if res_blocks_specs is None:
            # Default: one 3D block per level
            self.res_blocks_specs = [[True] for _ in range(depth+1)]
        else:
            assert_depth_args(res_blocks_specs)
            assert all(isinstance(itm, list) for itm in res_blocks_specs)
            self.res_blocks_specs = res_blocks_specs
        # Same for the decoder:
        if res_blocks_specs_decoder is None:
            # In this case copy setup of the encoder:
            self.res_blocks_specs_decoder = self.res_blocks_specs
        else:
            assert_depth_args(res_blocks_specs_decoder)
            assert all(isinstance(itm, list) for itm in res_blocks_specs_decoder)
            self.res_blocks_specs_decoder = res_blocks_specs_decoder

        # Parse decoder crops:
        self.decoder_crops = decoder_crops if decoder_crops is not None else {}
        assert len(self.decoder_crops) <= depth, "For the moment maximum one crop is supported"

        # Build the skeleton:
        super(MultiScaleInputsUNet3D, self).__init__(depth)

        # Build patchNets:
        global_kwargs = patchNet_kwargs.pop("global", {})
        self.nb_patch_nets = nb_patch_nets = len(patchNet_kwargs.keys())
        self.ptch_kwargs = [deepcopy(global_kwargs) for _ in range(nb_patch_nets)]
        for i in range(nb_patch_nets):
            self.ptch_kwargs[i].update(patchNet_kwargs[i])
        self.patch_models = nn.ModuleList([
            PatchNet(**kwgs) for kwgs in self.ptch_kwargs
        ])

        # Build embedding heads:
        emb_heads = {}
        for i in range(nb_patch_nets):
            depth_patch_net = patchNet_kwargs[i].get("depth_level", 0)
            emb_heads[depth_patch_net] = [] if depth_patch_net not in emb_heads else emb_heads[depth_patch_net]
            emb_heads[depth_patch_net].append(
                self.construct_embedding_heads(depth_patch_net, nb_patch_net=i))

        self.emb_heads = nn.ModuleDict(
            {str(dpth): nn.ModuleList(emb_heads[dpth]) for dpth in emb_heads}
        )

        self.autopad_first_encoding = AutoPad() if number_multiscale_inputs > 1 else None

        self.add_foreground_prediction_module = add_foreground_prediction_module
        self.foreground_prediction_kwargs = foreground_prediction_kwargs
        self.foreground_module = None
        if add_foreground_prediction_module:
            if self.foreground_prediction_kwargs is None:
                self.foreground_module = ConvNormActivation(self.decoder_fmaps[0],
                                          out_channels=1,
                                          kernel_size=1,
                                          dim=3,
                                          activation='Sigmoid',
                                          normalization=None,
                                          num_groups_norm=16)
            else:
                frg_kwargs = foreground_prediction_kwargs
                foreground_modules = {}
                for depth in frg_kwargs:
                    assert "nb_target" in frg_kwargs[depth]
                    foreground_modules[str(depth)] = ConvNormActivation(self.decoder_fmaps[depth],
                                           out_channels=1,
                                           kernel_size=1,
                                           dim=3,
                                           activation='Sigmoid',
                                           normalization=None,
                                           num_groups_norm=16)

                self.foreground_module = nn.ModuleDict(foreground_modules)



    @property
    def dim(self):
        return 3

    def concatenate_res_blocks(self, f_in, f_out, blocks_spec, depth):
        """
        Concatenate multiple residual blocks according to the config file
        """
        # TODO: generalize
        assert f_out % 16 == 0, "Not divisible for group norm!"

        blocks_list = []
        # Concatenate possible additional blocks:
        # TODO: get a default single block
        for is_3D in blocks_spec:
            assert isinstance(is_3D, bool)
            if is_3D:
                blocks_list.append(ResBlockAdvanced(f_in, f_inner=f_out,
                                 pre_kernel_size=(3,3,3),
                                 inner_kernel_size=(3, 3, 3),
                                 activation="ReLU",
                                 normalization="GroupNorm",
                                 num_groups_norm=16,
                                add_final_conv=False,
                                legacy_version=False
                                 ))
            else:
                blocks_list.append(ResBlockAdvanced(f_in, f_inner=f_out,
                                 pre_kernel_size=(1, 3, 3),
                                 inner_kernel_size=(1, 3, 3),
                                 activation="ReLU",
                                 normalization="GroupNorm",
                                 num_groups_norm=16,
                                 ))
            f_in = f_out

        return nn.Sequential(*blocks_list)


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

        sampler = UpsampleAndCrop(scale_factor=scale_factor, mode=self.upsampling_mode,
                                  crop_slice=self.decoder_crops.get(depth+1, None))

        return nn.Sequential(conv, sampler)

    def construct_decoder_module(self, depth):
        f_in = self.decoder_fmaps[depth]
        f_out = self.decoder_fmaps[depth]

        # Build blocks:
        blocks_spec = deepcopy(self.res_blocks_specs_decoder[depth])
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

    def construct_base_module(self):
        f_in = self.encoder_fmaps[self.depth - 1]
        f_out = self.encoder_fmaps[self.depth]
        blocks_spec = deepcopy(self.res_blocks_specs[self.depth])
        return self.concatenate_res_blocks(f_in, f_out, blocks_spec, self.depth)

    def construct_emb_slice(self, depth, previous_emb_slices):
        return (slice(None), slice(None))

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


    def forward(self, *inputs):
        nb_inputs = len(inputs)
        assert nb_inputs == self.number_multiscale_inputs, "Only two inputs accepted for the moment"

        encoded_states = []
        current = inputs[0]
        for encode, downsample, depth in zip(self.encoder_modules, self.downsampling_modules,
                                      range(self.depth)):
            if depth > 0 and depth < self.number_multiscale_inputs:
                current_lvl_padded = self.autopad_first_encoding(current, inputs[depth].shape)
                current = torch.cat((current_lvl_padded, inputs[depth]), dim=1)
                current = encode(current)
            else:
                current = encode(current)
            encoded_states.append(current)
            current = downsample(current)
        current = self.base_module(current)

        emb_outputs = []
        forgr_outputs = []
        for encoded_state, upsample, skip, merge, decode, depth in reversed(list(zip(
                encoded_states, self.upsampling_modules, self.skip_modules, self.merge_modules, self.decoder_modules, range(len(self.decoder_modules))))):
            current = upsample(current)
            current = merge(current, encoded_state)
            current = decode(current)

            if str(depth) in self.emb_heads:
                # We use reversed to make sure that they are outputed in the right order:
                all_emb_heads = reversed(self.emb_heads[str(depth)])
                for emb_head in all_emb_heads:
                    emb_out = emb_head(current)
                    emb_outputs.append(emb_out)

            if isinstance(self.foreground_module, nn.ModuleDict):
                if str(depth) in self.foreground_module:
                    forgr_outputs.append(self.foreground_module[str(depth)](current))
            elif depth == 0:
                forgr_outputs.append(self.foreground_module(current))

        emb_outputs.reverse()
        forgr_outputs.reverse()
        emb_outputs = emb_outputs + forgr_outputs

        if self.return_input:
            emb_outputs = emb_outputs + list(inputs)

        return emb_outputs

    def construct_merge_module(self, depth):
        return MergePyramidAndAutoCrop(self.decoder_fmaps[depth], self.encoder_fmaps[depth])


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
            apply_final_act = ASPP_kwargs.pop("apply_final_act", True)
            final_act = ASPP_kwargs.get("final_act", "ReLU") if apply_final_act else None
            emb_head = ConvNormActivation(self.decoder_fmaps[depth],
                                             out_channels=ptch_kwargs["latent_variable_size"],
                                             kernel_size=1,
                                             dim=3,
                                             activation=final_act,
                                             normalization=norm,
                                             num_groups_norm=16)

        crop = self.decoder_crops.get(depth, None)
        emb_head = nn.Sequential(emb_head, Crop(crop)) if crop is not None else emb_head

        return emb_head

    def construct_encoder_module(self, depth):
        if depth == 0:
            f_in = self.in_channels
        elif depth < self.number_multiscale_inputs:
            f_in = self.encoder_fmaps[depth - 1] + self.in_channels
        else:
            f_in = self.encoder_fmaps[depth - 1]
        f_out = self.encoder_fmaps[depth]

        # Build blocks:
        blocks_spec = deepcopy(self.res_blocks_specs[depth])

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

        # Build models (now PatchNets are also automatically built here):
        if type_of_model == "MultiScaleInputsUNet3D":
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



class PredictForegroundFromStackedModel(GeneralizedStackedPyramidUNet3D):
    # FIXME: should be a subclass of UNetMultiInput
    def __init__(self,
                 keep_only_first_foreground=True,
                 **super_kwargs):
        super(PredictForegroundFromStackedModel, self).__init__(**super_kwargs)

        last_model = self.models[-1]

        assert last_model.foreground_module is not None
        if isinstance(last_model.foreground_module, nn.ModuleDict):
            nb_foreground_pred = len(last_model.foreground_module)
        else :
            nb_foreground_pred = 1

        assert isinstance(last_model.emb_heads, nn.ModuleDict)
        nb_emb_pred = 0
        for depth in last_model.emb_heads:
            nb_emb_pred += len(last_model.emb_heads[depth])

        self.nb_emb_pred = nb_emb_pred
        self.keep_only_first_foreground = keep_only_first_foreground
        self.nb_foreground_pred = nb_foreground_pred
        self.nb_raw = last_model.number_multiscale_inputs if last_model.return_input else 0

        # Freeze parameters model:
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, *inputs):
        with torch.no_grad():
            predictions = super(PredictForegroundFromStackedModel, self).forward(*inputs)

        foreground_preds = predictions[self.nb_emb_pred:]
        if self.keep_only_first_foreground:
            foreground_preds = foreground_preds[0]
        else:
            if self.nb_raw != 0:
                foreground_preds = foreground_preds[:-self.nb_raw]

        return foreground_preds



class AffinitiesFromEmb(nn.Module):
    def __init__(self, path_backbone,
                 nb_offsets,
                 prediction_indices,
                 final_layer_kwargs=None,
                 train_backbone=False,
                 reload_backbone=True,
                 nb_extra_final_layers=0,
                 nb_channels_final_layers=None,
                 nb_channels_output_emb=None,
                 **stacked_model_super_kwargs):
        super(AffinitiesFromEmb, self).__init__()

        self.path_backbone = path_backbone
        stacked_model_super_kwargs.pop('slicing_config', None)
        self.stacked_model_super_kwargs = stacked_model_super_kwargs
        self.train_backbone = train_backbone
        self.reload_backbone = reload_backbone
        self.load_backbone()


        self.prediction_indices = prediction_indices
        raise DeprecationWarning("output_fmaps was deleted")
        output_maps = nb_channels_output_emb if nb_channels_output_emb is not None else \
            self.backbone_model.models[-1].output_fmaps * len(prediction_indices)
        nb_channels_final_layers = nb_channels_final_layers if nb_channels_final_layers is not None else int(output_maps/2)

        # Build some 1x1 layers:
        layers = [ConvNormActivation(in_channels=output_maps,
                                     out_channels=nb_channels_final_layers,
                                     **final_layer_kwargs)]
        for _ in range(nb_extra_final_layers):
            layers.append(ConvNormActivation(in_channels=nb_channels_final_layers,
                                             out_channels=nb_channels_final_layers,
                                             **final_layer_kwargs))

        layers.append(ConvNormActivation(in_channels=nb_channels_final_layers,
                                         out_channels=nb_offsets,
                                         kernel_size=(1,1,1),
                                         dim=3,
                                         activation="Sigmoid",
                                         normalization=None))
        self.final_module = nn.Sequential(*layers)

    def forward(self, *inputs):
        with torch.no_grad():
            current = self.backbone_model(*inputs)

        # Concatenate predictions:
        current = [current[idx] for idx in self.prediction_indices]
        current = torch.cat(current, dim=1)

        current = self.final_module(current)
        return current

    def load_state_dict(self, state_dict):
        super(AffinitiesFromEmb, self).load_state_dict(state_dict)
        # Reload backbone:
        if self.reload_backbone:
            self.load_backbone()

    def load_backbone(self):
        self.backbone_model = GeneralizedStackedPyramidUNet3D(**self.stacked_model_super_kwargs)
        state_dict = torch.load(self.path_backbone)["_model"].state_dict()
        self.backbone_model.load_state_dict(state_dict)

        # Freeze-parameters for models that are not trained:
        if not self.train_backbone:
            print("Backbone parameters freezed!")
            for param in self.backbone_model.parameters():
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
        raise DeprecationWarning("output_fmaps was deleted")
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
