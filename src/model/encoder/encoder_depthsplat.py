from dataclasses import dataclass
from typing import Literal, Optional, List

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder
from .visualization.encoder_visualizer_depthsplat_cfg import EncoderVisualizerDepthSplatCfg

import torchvision.transforms as T
import torch.nn.functional as F

from .unimatch.mv_unimatch import MultiViewUniMatch, set_num_views
from .unimatch.ldm_unet.unet import UNetModel
from .unimatch.feature_upsampler import ResizeConvFeatureUpsampler


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int
    no_mapping: bool


@dataclass
class EncoderDepthSplatCfg:
    name: Literal["depthsplat"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    visualizer: EncoderVisualizerDepthSplatCfg
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    unimatch_weights_path: str | None
    downscale_factor: int
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]

    # mv_unimatch
    num_scales: int
    upsample_factor: int
    lowest_feature_resolution: int
    depth_unet_channels: int
    grid_sample_disable_cudnn: bool

    # depthsplat color branch
    large_gaussian_head: bool
    color_large_unet: bool
    init_sh_input_img: bool
    feature_upsampler_channels: int
    gaussian_regressor_channels: int

    # loss config
    supervise_intermediate_depth: bool
    return_depth: bool

    # only depth
    train_depth_only: bool

    # monodepth config
    monodepth_vit_type: str

    # offset_mode
    offset_mode: str 
    offset_factor: float

    # confidence activation
    confidence_activation: str | None

    # multi-view matching
    costvolume_nearest_n_views: Optional[int] = None
    multiview_trans_nearest_n_views: Optional[int] = None


class EncoderDepthSplat(Encoder[EncoderDepthSplatCfg]):
    def __init__(self, cfg: EncoderDepthSplatCfg) -> None:
        super().__init__(cfg)

        self.depth_predictor = MultiViewUniMatch(
            num_scales=cfg.num_scales,
            upsample_factor=cfg.upsample_factor,
            lowest_feature_resolution=cfg.lowest_feature_resolution,
            vit_type=cfg.monodepth_vit_type,
            unet_channels=cfg.depth_unet_channels,
            grid_sample_disable_cudnn=cfg.grid_sample_disable_cudnn,
        )

        if self.cfg.train_depth_only:
            return

        # upsample to the original resolution
        self.feature_upsampler = ResizeConvFeatureUpsampler(num_scales=cfg.num_scales,
                                                            lowest_feature_resolution=cfg.lowest_feature_resolution,
                                                            out_channels=self.cfg.feature_upsampler_channels,
                                                            vit_type=self.cfg.monodepth_vit_type,
                                                            )
        feature_upsampler_channels = self.cfg.feature_upsampler_channels
        
        # gaussians adapter
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        # unet
        # concat(img, depth, match_prob, features)
        in_channels = 3 + 1 + 1 + feature_upsampler_channels
        channels = self.cfg.gaussian_regressor_channels

        modules = [
            nn.Conv2d(in_channels, channels, 3, 1, 1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        ]

        if self.cfg.color_large_unet or self.cfg.gaussian_regressor_channels == 16:
            unet_channel_mult = [1, 2, 4, 4, 4]
        else:
            unet_channel_mult = [1, 1, 1, 1, 1]
        unet_attn_resolutions = [16]

        modules.append(
            UNetModel(
                image_size=None,
                in_channels=channels,
                model_channels=channels,
                out_channels=channels,
                num_res_blocks=1,
                attention_resolutions=unet_attn_resolutions,
                channel_mult=unet_channel_mult,
                num_head_channels=32 if self.cfg.gaussian_regressor_channels >= 32 else 16,
                dims=2,
                postnorm=False,
                num_frames=2,
                use_cross_view_self_attn=True,
            )
        )

        modules.append(nn.Conv2d(channels, channels, 3, 1, 1))

        self.gaussian_regressor = nn.Sequential(*modules)
        
        # predict gaussian parameters: scale, q, sh
        num_gaussian_parameters = self.gaussian_adapter.d_in + 2
        if self.cfg.confidence_activation is not None:
            # predict opacity and confidence
            num_gaussian_parameters += 2
        else:
            # predict opacity
            num_gaussian_parameters += 1

        # concat(img, features, unet_out, match_prob)
        in_channels = 3 + feature_upsampler_channels + channels + 1

        if self.cfg.feature_upsampler_channels != 128:
            self.gaussian_head = nn.Sequential(
                nn.Conv2d(in_channels, num_gaussian_parameters,
                            3, 1, 1, padding_mode='replicate'),
                nn.GELU(),
                nn.Conv2d(num_gaussian_parameters,
                            num_gaussian_parameters, 3, 1, 1, padding_mode='replicate')
            )
        else:
            self.gaussian_head = nn.Sequential(
                nn.Conv2d(
                    in_channels, num_gaussian_parameters * 2, 3, 1, 1, padding_mode='replicate'),
                nn.GELU(),
                nn.Conv2d(num_gaussian_parameters * 2,
                            num_gaussian_parameters, 3, 1, 1, padding_mode='replicate')
            )

        if self.cfg.init_sh_input_img:
            nn.init.zeros_(self.gaussian_head[-1].weight[10:])
            nn.init.zeros_(self.gaussian_head[-1].bias[10:])

        # init scale
        # first 3: opacity, confidence, offset_xy
        nn.init.zeros_(self.gaussian_head[-1].weight[3:6])
        nn.init.zeros_(self.gaussian_head[-1].bias[3:6])
        nn.init.constant_(self.gaussian_head[-1].bias[1], 0.5) 

    def load_state_dict(self, state_dict, strict=False):
        new_state_dict = {}
        
        for k, v in state_dict.items():
            if 'gaussian_head' in k:
                continue
            
            new_state_dict[k] = v
        
        has_gaussian_head = any('gaussian_head.0.weight' in k for k in state_dict.keys())
        
        if has_gaussian_head:
            current_shapes = {
                'head_0_weight': self.gaussian_head[0].weight.shape,
                'head_0_bias': self.gaussian_head[0].bias.shape,
                'head_2_weight': self.gaussian_head[2].weight.shape,
                'head_2_bias': self.gaussian_head[2].bias.shape
            }
            
            pretrained_shapes = {
                'head_0_weight': state_dict['gaussian_head.0.weight'].shape,
                'head_0_bias': state_dict['gaussian_head.0.bias'].shape,
                'head_2_weight': state_dict['gaussian_head.2.weight'].shape,
                'head_2_bias': state_dict['gaussian_head.2.bias'].shape
            }
            shapes_match = all(
                current_shapes[k] == pretrained_shapes[k] 
                for k in current_shapes.keys()
            )
            
            if shapes_match:
                print("Gaussian Head Shape match! Loading original weights...")
                new_state_dict['gaussian_head.0.weight'] = state_dict['gaussian_head.0.weight']
                new_state_dict['gaussian_head.0.bias'] = state_dict['gaussian_head.0.bias']
                new_state_dict['gaussian_head.2.weight'] = state_dict['gaussian_head.2.weight']
                new_state_dict['gaussian_head.2.bias'] = state_dict['gaussian_head.2.bias']

                if 'gaussian_head.1.weight' in state_dict:
                    new_state_dict['gaussian_head.1.weight'] = state_dict['gaussian_head.1.weight']
                if 'gaussian_head.1.bias' in state_dict:
                    new_state_dict['gaussian_head.1.bias'] = state_dict['gaussian_head.1.bias']
            else:

                print(f"Current shapes: {current_shapes}")
                print(f"Pretrained shapes: {pretrained_shapes}")

                pretrained_weight = state_dict['gaussian_head.0.weight']
                new_weight = torch.zeros(current_shapes['head_0_weight'], 
                                        dtype=pretrained_weight.dtype, 
                                        device=pretrained_weight.device)
                

                min_out_channels = min(pretrained_shapes['head_0_weight'][0], current_shapes['head_0_weight'][0])
                min_in_channels = min(pretrained_shapes['head_0_weight'][1], current_shapes['head_0_weight'][1])
                
                new_weight[:min_out_channels, :min_in_channels] = pretrained_weight[:min_out_channels, :min_in_channels]

                if current_shapes['head_0_weight'][0] > pretrained_shapes['head_0_weight'][0]:
                    extra_out_channels = current_shapes['head_0_weight'][0] - pretrained_shapes['head_0_weight'][0]
                    channel_mean = pretrained_weight.mean(dim=0, keepdim=True)
                    new_weight[min_out_channels:, :min_in_channels] = channel_mean.repeat(extra_out_channels, 1, 1, 1)[:, :min_in_channels]
                
                if current_shapes['head_0_weight'][1] > pretrained_shapes['head_0_weight'][1]:
                    in_channel_mean = pretrained_weight[:min_out_channels].mean(dim=1, keepdim=True)
                    new_weight[:min_out_channels, min_in_channels:] = in_channel_mean.repeat(1, current_shapes['head_0_weight'][1] - min_in_channels, 1, 1)
                
                if current_shapes['head_0_weight'][0] > pretrained_shapes['head_0_weight'][0] and current_shapes['head_0_weight'][1] > pretrained_shapes['head_0_weight'][1]:
                    overall_mean = pretrained_weight.mean()
                    new_weight[min_out_channels:, min_in_channels:] = overall_mean
                
                new_state_dict['gaussian_head.0.weight'] = new_weight
                
                pretrained_bias = state_dict['gaussian_head.0.bias']
                new_bias = torch.zeros(current_shapes['head_0_bias'], 
                                    dtype=pretrained_bias.dtype, 
                                    device=pretrained_bias.device)
                
                min_size = min(pretrained_shapes['head_0_bias'][0], current_shapes['head_0_bias'][0])
                new_bias[:min_size] = pretrained_bias[:min_size]
                
                if current_shapes['head_0_bias'][0] > pretrained_shapes['head_0_bias'][0]:
                    bias_mean = pretrained_bias.mean()
                    new_bias[min_size:] = bias_mean
                
                new_state_dict['gaussian_head.0.bias'] = new_bias
                
                pretrained_weight = state_dict['gaussian_head.2.weight']
                new_weight = torch.zeros(current_shapes['head_2_weight'], 
                                        dtype=pretrained_weight.dtype, 
                                        device=pretrained_weight.device)

                out_channels_old = pretrained_shapes['head_2_weight'][0]
                in_channels_old = pretrained_shapes['head_2_weight'][1]
                out_channels_new = current_shapes['head_2_weight'][0]
                in_channels_new = current_shapes['head_2_weight'][1]
                
                if out_channels_old > 0 and out_channels_new > 0:
                    min_in_channels = min(in_channels_old, in_channels_new)
                    new_weight[0, :min_in_channels] = pretrained_weight[0, :min_in_channels]
                    
                    if in_channels_new > in_channels_old:
                        in_mean = pretrained_weight[0].mean(dim=0, keepdim=True)
                        new_weight[0, min_in_channels:] = in_mean.repeat(in_channels_new - min_in_channels, 1, 1)
                
                if out_channels_old > 0 and out_channels_new > 1:
                    new_weight[1, :] = 0.0 
                
                src_remaining = out_channels_old - 1 
                dst_remaining = out_channels_new - 2 
                min_channels = min(src_remaining, dst_remaining)
                
                if min_channels > 0:
                    min_in_channels = min(in_channels_old, in_channels_new)
                    new_weight[2:2+min_channels, :min_in_channels] = pretrained_weight[1:1+min_channels, :min_in_channels]

                    if in_channels_new > in_channels_old:
                        for i in range(min_channels):
                            in_mean = pretrained_weight[1+i].mean(dim=0, keepdim=True)
                            new_weight[2+i, min_in_channels:] = in_mean.repeat(in_channels_new - min_in_channels, 1, 1)

                if dst_remaining > src_remaining:
                    if src_remaining > 0:
                        channel_mean = pretrained_weight[1:1+src_remaining].mean(dim=0, keepdim=True)
                    else:
                        channel_mean = pretrained_weight[0:1].mean(dim=0, keepdim=True)
                    
                    new_weight[2+min_channels:] = channel_mean.repeat(dst_remaining - src_remaining, 1, 1, 1)
                
                new_state_dict['gaussian_head.2.weight'] = new_weight

                pretrained_bias = state_dict['gaussian_head.2.bias']
                new_bias = torch.zeros(current_shapes['head_2_bias'], 
                                    dtype=pretrained_bias.dtype, 
                                    device=pretrained_bias.device)

                if 0 < pretrained_shapes['head_2_bias'][0] and 0 < current_shapes['head_2_bias'][0]:
                    new_bias[0] = pretrained_bias[0]

                if 1 < current_shapes['head_2_bias'][0]:
                    new_bias[1] = 0.5 

                src_remaining = pretrained_shapes['head_2_bias'][0] - 1 
                dst_remaining = current_shapes['head_2_bias'][0] - 2  
                min_channels = min(src_remaining, dst_remaining)

                if min_channels > 0:
                    new_bias[2:2+min_channels] = pretrained_bias[1:1+min_channels]
                
                if dst_remaining > src_remaining:
                    if src_remaining > 0:
                        bias_mean = pretrained_bias[1:1+src_remaining].mean()
                    else:
                        bias_mean = pretrained_bias[0]
                
                    new_bias[2+min_channels:] = bias_mean
                
                new_state_dict['gaussian_head.2.bias'] = new_bias
                
                if 'gaussian_head.1.weight' in state_dict:
                    new_state_dict['gaussian_head.1.weight'] = state_dict['gaussian_head.1.weight']
                if 'gaussian_head.1.bias' in state_dict:
                    new_state_dict['gaussian_head.1.bias'] = state_dict['gaussian_head.1.bias']
        
        result = super().load_state_dict(new_state_dict, strict=False)
        print(f"Succeeded! missing: {len(result.missing_keys)}, unexpected: {len(result.unexpected_keys)}")
        return result 

    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
    ):
        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        if (
            self.cfg.costvolume_nearest_n_views is not None
            or self.cfg.multiview_trans_nearest_n_views is not None
        ):
            assert self.cfg.costvolume_nearest_n_views is not None
            with torch.no_grad():
                xyzs = context["extrinsics"][:, :, :3, -1].detach()
                cameras_dist_matrix = torch.cdist(xyzs, xyzs, p=2)
                cameras_dist_index = torch.argsort(cameras_dist_matrix)

                cameras_dist_index = cameras_dist_index[:,
                                                        :, :self.cfg.costvolume_nearest_n_views]
        else:
            cameras_dist_index = None

        # depth prediction
        results_dict = self.depth_predictor(
            context["image"],
            attn_splits_list=[2],
            min_depth=1. / context["far"],
            max_depth=1. / context["near"],
            intrinsics=context["intrinsics"],
            extrinsics=context["extrinsics"],
            nn_matrix=cameras_dist_index,
        )

        # list of [B, V, H, W], with all the intermediate depths
        depth_preds = results_dict['depth_preds']

        # [B, V, H, W]
        depth = depth_preds[-1]

        if self.cfg.train_depth_only:
            # convert format
            # [B, V, H*W, 1, 1]
            depths = rearrange(depth, "b v h w -> b v (h w) () ()")

            if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:
                # supervise all the intermediate depth predictions
                num_depths = len(depth_preds)

                # [B, V, H*W, 1, 1]
                intermediate_depths = torch.cat(
                    depth_preds[:(num_depths - 1)], dim=0)
                intermediate_depths = rearrange(
                    intermediate_depths, "b v h w -> b v (h w) () ()")

                # concat in the batch dim
                depths = torch.cat((intermediate_depths, depths), dim=0)

                b *= num_depths

            # return depth prediction for supervision
            depths = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            ).squeeze(-1).squeeze(-1)
            # print(depths.shape)  # [B, V, H, W]

            return {
                "gaussians": None,
                "depths": depths
            }

        # update the num_views in unet attention, useful for random input views
        set_num_views(self.gaussian_regressor, v)

        # features [BV, C, H, W]
        features = self.feature_upsampler(results_dict["features_cnn"],
                                            results_dict["features_mv"],
                                            results_dict["features_mono"],
                                            )

        # match prob from softmax
        # [BV, D, H, W] in feature resolution
        match_prob = results_dict['match_probs'][-1]
        match_prob = torch.max(match_prob, dim=1, keepdim=True)[
            0]  # [BV, 1, H, W]
        match_prob = F.interpolate(
            match_prob, size=depth.shape[-2:], mode='nearest')

        # unet input
        concat = torch.cat((
            rearrange(context["image"], "b v c h w -> (b v) c h w"),
            rearrange(depth, "b v h w -> (b v) () h w"),
            match_prob,
            features,
        ), dim=1)

        out = self.gaussian_regressor(concat)

        concat = [out,
                    rearrange(context["image"],
                            "b v c h w -> (b v) c h w"),
                    features,
                    match_prob]

        out = torch.cat(concat, dim=1)

        gaussians = self.gaussian_head(out)  # [BV, C, H, W]

        gaussians = rearrange(gaussians, "(b v) c h w -> b v c h w", b=b, v=v)

        depths = rearrange(depth, "b v h w -> b v (h w) () ()")

        # [B, V, H*W, 1, 1]
        densities = rearrange(
            match_prob, "(b v) c h w -> b v (c h w) () ()", b=b, v=v)
        # [B, V, H*W, 84]
        raw_gaussians = rearrange(
            gaussians, "b v c h w -> b v (h w) c")

        if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:

            # supervise all the intermediate depth predictions
            num_depths = len(depth_preds)

            # [B, V, H*W, 1, 1]
            intermediate_depths = torch.cat(
                depth_preds[:(num_depths - 1)], dim=0)
            
            intermediate_depths = rearrange(
                intermediate_depths, "b v h w -> b v (h w) () ()")

            # concat in the batch dim
            depths = torch.cat((intermediate_depths, depths), dim=0)

            # shared color head
            densities = torch.cat([densities] * num_depths, dim=0)
            raw_gaussians = torch.cat(
                [raw_gaussians] * num_depths, dim=0)

            b *= num_depths

        # [B, V, H*W, 1, 1]
        opacities = raw_gaussians[..., :1].sigmoid().unsqueeze(-1)

        if self.cfg.confidence_activation is None:
            raw_gaussians = raw_gaussians[..., 1:]
            confidences = None
        elif self.cfg.confidence_activation == "softplus":
            confidences = F.softplus(raw_gaussians[..., 1:2]).unsqueeze(-1)
            confidences = confidences + densities.detach()
            raw_gaussians = raw_gaussians[..., 2:]
        elif self.cfg.confidence_activation == "sigmoid":
            confidences = torch.sigmoid(raw_gaussians[..., 1:2]).unsqueeze(-1)
            confidences = confidences + densities.detach()
            raw_gaussians = raw_gaussians[..., 2:]
        else:
            raise ValueError(
                f"Unknown confidence activation: {self.cfg.confidence_activation}"
            )
        
        # Convert the features and depths into Gaussians.
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=self.cfg.num_surfaces,
        )
        if self.cfg.offset_mode == "pixel":
            # Offset in the range of [-max_offset/2, max_offset/2] pixels
            max_offset = self.cfg.offset_factor 
            offset_xy = torch.tanh(gaussians[..., :2]) * max_offset / 2 
            pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
            xy_ray = xy_ray + offset_xy * pixel_size
        elif self.cfg.offset_mode == "unconstrained":
            # Without any activation, make the position more flexible
            zoom_factor = self.cfg.offset_factor
            offset_xy = gaussians[..., :2] * zoom_factor
            # Since the chamfer loss will constrain the offset, no need to limit the range here
            pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
            xy_ray = xy_ray + offset_xy * pixel_size
        elif self.cfg.offset_mode == "ori" or self.cfg.offset_mode == "origin":
            # Offset in the range of [0, 1] pixel
            offset_xy = gaussians[..., :2].sigmoid()
            pixel_size = 1 / \
                torch.tensor((w, h), dtype=torch.float32, device=device)
            xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
        else:
            print("Unknown offset mode")
            raise ValueError(f"Unknown offset mode: {self.cfg.offset_mode}")

        sh_input_images = context["image"]

        if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:
            context_extrinsics = torch.cat(
                [context["extrinsics"]] * len(depth_preds), dim=0)
            context_intrinsics = torch.cat(
                [context["intrinsics"]] * len(depth_preds), dim=0)

            gaussians = self.gaussian_adapter.forward(
                rearrange(context_extrinsics, "b v i j -> b v () () () i j"),
                rearrange(context_intrinsics, "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                depths,
                opacities,
                rearrange(
                    gaussians[..., 2:],
                    "b v r srf c -> b v r srf () c",
                ),
                (h, w),
                input_images=sh_input_images.repeat(
                    len(depth_preds), 1, 1, 1, 1) if self.cfg.init_sh_input_img else None,
                confidences=confidences,
            )

        else:
            gaussians = self.gaussian_adapter.forward(
                rearrange(context["extrinsics"],
                          "b v i j -> b v () () () i j"),
                rearrange(context["intrinsics"],
                          "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                depths,
                opacities,
                rearrange(
                    gaussians[..., 2:],
                    "b v r srf c -> b v r srf () c",
                ),
                (h, w),
                input_images=sh_input_images if self.cfg.init_sh_input_img else None,
                confidences=confidences,
            )

        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )

        gaussians = Gaussians(
            rearrange(
                gaussians.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),

            rearrange(
                gaussians.confidences,
                "b v r srf spp -> b (v r srf spp)",
            ) if gaussians.confidences is not None else None,
        )

        if self.cfg.return_depth:
            # return depth prediction for supervision
            depths = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            ).squeeze(-1).squeeze(-1)
            # print(depths.shape)  # [B, V, H, W]

            return {
                "gaussians": gaussians,
                "depths": depths
            }

        return gaussians

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size
                * self.cfg.downscale_factor,
            )

            return batch

        return data_shim

    @property
    def sampler(self):
        return None
