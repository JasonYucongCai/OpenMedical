"""
ct_model.model — ConvNeXt-Tiny Chest X-ray model definition
=============================================================
v5 Industrial: ConvNeXt-Tiny + ASL + cRT | 512px | 14 pathology classes
Based on Sulake 2026 (ISBI 5th/68) + Hong 2023 CXR-LT + Ben-Baruch ICCV 2021 ASL.
"""

import torch.nn as nn
from torchvision import models

# ---- Constants ----
PATHOLOGY_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax"
]
NUM_CLASSES = len(PATHOLOGY_LABELS)


class ConvNeXtChestXray(nn.Module):
    """
    ConvNeXt-Tiny backbone with custom classifier head for 14 chest pathologies.

    Architecture:
        - ConvNeXt-Tiny (ImageNet pretrained) as feature extractor
        - Custom classifier: Dropout → Linear(768→512) → LayerNorm → GELU
          → Dropout → Linear(512→14)

    Args:
        num_classes: Number of output classes (default 14 for NIH ChestX-ray14)
        pretrained:  Use ImageNet pretrained weights
        dropout:     Dropout probability in classifier head
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.backbone = models.convnext_tiny(
            weights="IMAGENET1K_V1" if pretrained else None
        )
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        """Freeze all backbone parameters; only classifier remains trainable."""
        for name, p in self.named_parameters():
            if "classifier" not in name:
                p.requires_grad = False

    def get_classifier_params(self):
        """Get classifier parameters for separate optimization (cRT stage)."""
        return [p for n, p in self.named_parameters() if "classifier" in n]
