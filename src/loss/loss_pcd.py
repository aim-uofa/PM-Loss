from dataclasses import dataclass
import torch
from jaxtyping import Float
from torch import Tensor

from pmloss import PMLoss 

from ..dataset.types import BatchedExample
from ..model.types import Gaussians
from .loss import Loss
from ..misc.nn_module_tools import convert_to_buffer

@dataclass
class LossPcdCfg:
    weight: float
    apply_after_step: int
    gt_mode: str = "vggt"
    loss_mode: str = "3d"
    ignore_large_loss: float = 100.0

@dataclass
class LossPcdCfgWrapper:
    pcd: LossPcdCfg

class LossPcd(Loss[LossPcdCfg, LossPcdCfgWrapper]):
    pm_loss: PMLoss

    def __init__(self, cfg: LossPcdCfgWrapper) -> None:
        super().__init__(cfg)
        # Initialize the reusable PM-Loss metric
        self.pm_loss = PMLoss()

        convert_to_buffer(self.pm_loss, persistent=False)
    
    def forward(
        self,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        # Before the specified step, don't apply the loss.
        if global_step < self.cfg.apply_after_step:
            return torch.tensor(0.0, device=batch["context"]["image"].device)
        
        # Extract the required inputs
        context_images = batch["context"]["image"]
        predict_points = gaussians.means[-context_images.shape[0]:]
        

        # Call the PM-Loss metric
        loss = self.pm_loss(
            predict_points=predict_points,
            context_images=context_images,
            loss_mode=self.cfg.loss_mode,
            ignore_large_loss=self.cfg.ignore_large_loss
        )
        
        # Apply weighting and reduce the batch loss
        return self.cfg.weight * loss.mean()