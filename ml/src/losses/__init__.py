from .bce_dice import BCEDiceLoss
from .gdl import GeneralizedDiceLoss
from .ssim import ssim_loss
from .sliced_wasserstein import sliced_wasserstein_distance

__all__ = ["BCEDiceLoss", "GeneralizedDiceLoss", "ssim_loss", "sliced_wasserstein_distance"]
