import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from einops import rearrange
from pytorch3d.loss import chamfer_distance
from .vggt.models.vggt import VGGT
from .utils import align_point_clouds_umeyama

class PMLoss(nn.Module):
    """
    Point Metric Loss (PM-Loss)
    
    Calculates a point cloud-based loss by generating a pseudo-ground truth
    point cloud from context images using a pre-trained VGGT model and
    comparing it against a predicted point cloud.
    """
    def __init__(self, model: str = "vggt", device: str | torch.device = "cuda"):
        super().__init__()
        self.model = model.lower()
        if self.model == 'vggt':
            # Load the VGGT model from Hugging Face
            model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
            self.vggt = model.eval()

            # Freeze the model parameters as it's only used for inference
            for param in self.vggt.parameters():
                param.requires_grad = False
        else:
            raise ValueError(f"Model '{model}' not supported. Currently only 'vggt' is implemented.")
        
        # Determine the optimal dtype for inference
        major, _ = torch.cuda.get_device_capability()
        self.dtype = torch.bfloat16 if major >= 8 else torch.float16
        self.device = device

    def forward(
        self,
        predict_points: Float[Tensor, "b n 3"],
        context_images: Float[Tensor, "b v c h w"],
        loss_mode: str = "3d",
        ignore_large_loss: float = 100.0,
    ) -> Float[Tensor, "b"]:
        """
        Calculates the forward pass for the PM-Loss.

        Args:
            predict_points: The predicted point cloud batch (B, N, 3).
            context_images: The batch of context images (B, V, C, H, W).
            loss_mode: The loss mode to use. Currently only "3d" is supported.
            ignore_large_loss: Threshold to cap large loss values.

        Returns:
            A tensor of shape (B,) containing the loss for each item in the batch.
        """
        b, v, c, h, w = context_images.shape
        if self.model == 'vggt':
            # --- 1. Preprocess images for VGGT ---
            # VGGT expects input dimensions to be multiples of its patch size (14)
            H_ = (h // 14) * 14
            W_ = (w // 14) * 14
            
            if H_ != h or W_ != w:
                image_input_ = nn.functional.interpolate(
                    context_images.flatten(0, 1), 
                    (H_, W_), 
                    mode='bilinear',
                    align_corners=False
                ).view(b, v, c, H_, W_)
            else:
                image_input_ = context_images
            
            # --- 2. Generate pseudo-GT point cloud with VGGT ---
            with torch.cuda.amp.autocast(enabled=True, dtype=self.dtype):
                vggt_res = self.vggt(image_input_)
        
            world_points = vggt_res["world_points"]  # [b, v, H_, W_, 3]
            
            # --- 3. Postprocess pseudo-GT point cloud ---
            # Upsample the point map back to the original resolution if needed
            if H_ != h or W_ != w:
                world_points = rearrange(world_points, "b v h w c -> (b v) c h w")
                world_points = nn.functional.interpolate(
                    world_points, 
                    (h, w), 
                    mode='nearest'
                )
                world_points = rearrange(world_points, "(b v) c h w -> b v h w c", b=b, v=v)
            
            if loss_mode == "3d":
                gt_points = rearrange(world_points, "b v h w c -> b (v h w) c")
                # Align the ground truth points with the predicted points using Umeyama
                gt_points, _ = align_point_clouds_umeyama(gt_points, predict_points)
            else:
                raise ValueError(f"Loss mode '{loss_mode}' not supported.")

        # --- 4. Calculate Chamfer Distance ---
        loss, _ = chamfer_distance(
            predict_points, gt_points,
            single_directional=True,
            point_reduction="mean",
            batch_reduction=None,  # Return loss per batch item
            norm=2,
        )

        # --- 5. Stabilize loss ---
        if torch.isnan(loss).any():
            print("Warning: PM-Loss contains NaN values. Clamping to zero.")
            loss = torch.nan_to_num(loss, nan=0.0)

        # Ignore abnormally large losses to stabilize training
        loss = torch.where(loss > ignore_large_loss, loss * 0.00001, loss)
        
        return loss