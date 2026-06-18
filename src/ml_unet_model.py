import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import logging

log = logging.getLogger(__name__)

class ReefUNet(nn.Module):
    """
    U-Net Architecture for Reef Semantic Segmentation using SMP.
    """
    def __init__(self, encoder_name="resnet18", in_channels=1, classes=1):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet", # Use pre-trained weights for transfer learning
            in_channels=in_channels,    # 1 channel (Bathymetry)
            classes=classes,            # 1 class (Reef mask)
            activation=None             # We use BCEWithLogitsLoss which expects raw logits
        )

    def forward(self, x):
        return self.model(x)

def get_loss_function():
    """
    Combined BCE + Dice Loss for robust segmentation on imbalanced data (lots of sand, small reefs).
    """
    # Dice loss helps with sharp boundaries, BCE handles pixel-wise accuracy
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    
    def combined_loss(y_pred, y_true):
        return bce_loss(y_pred, y_true) + dice_loss(y_pred, y_true)
        
    return combined_loss

def calculate_iou(y_pred, y_true, threshold=0.5):
    """
    Calculate Intersection over Union (IoU) metric.
    """
    probs = torch.sigmoid(y_pred)
    preds = (probs > threshold).float()
    
    intersection = (preds * y_true).sum()
    union = preds.sum() + y_true.sum() - intersection
    
    if union == 0:
        return 1.0 # Both are empty
    return (intersection / union).item()
