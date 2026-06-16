import torch
import torch.nn as nn
import timm
from config import NUM_CLASSES


def build_model(
    efficientnet_version: str = "b0",
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Build an EfficientNet model for skin lesion classification.

    Args:
        efficientnet_version: One of 'b0', 'b1', 'b2', 'b3'.
        pretrained:           Load ImageNet pretrained weights.
        dropout:              Dropout rate before the final classifier.
        freeze_backbone:      If True, freeze all layers except the classifier.

    Returns:
        PyTorch model ready for training.
    """
    model_name = f"efficientnet_{efficientnet_version}"
    model = timm.create_model(model_name, pretrained=pretrained)

    # Get the input features of the original classifier
    in_features = model.classifier.in_features

    # Replace classifier with our custom head
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, NUM_CLASSES),
    )

    if freeze_backbone:
        freeze_backbone_layers(model)

    return model


def freeze_backbone_layers(model: nn.Module) -> None:
    """
    Freeze all layers except the classifier head.
    Used in Phase 1 of fine-tuning.
    """
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Frozen backbone — Trainable params: {trainable:,} / {total:,}")


def unfreeze_all_layers(model: nn.Module) -> None:
    """
    Unfreeze all layers for full fine-tuning.
    Used in Phase 2 of fine-tuning.
    """
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Unfrozen all layers — Trainable params: {trainable:,}")


def get_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Extract feature embeddings before the classifier head.
    Used later for the multimodal fusion branch.

    Args:
        model: EfficientNet model built with build_model().
        x:     Input tensor of shape (B, C, H, W).

    Returns:
        Feature tensor of shape (B, in_features).
    """
    # timm EfficientNet: forward_features → global pool → classifier
    features = model.forward_features(x)           # (B, C, H, W)
    features = model.global_pool(features)          # (B, in_features)
    if model.global_pool.flatten:
        features = features.flatten(1)
    return features


def get_model_info(efficientnet_version: str) -> dict:
    """
    Return basic info about the model variant.
    """
    input_sizes = {"b0": 224, "b1": 240, "b2": 260, "b3": 300}
    params_m    = {"b0": 5.3, "b1": 7.8, "b2": 9.2, "b3": 12.0}

    return {
        "version":    efficientnet_version,
        "input_size": input_sizes[efficientnet_version],
        "params_M":   params_m[efficientnet_version],
    }

class MetadataBranch(nn.Module):
    """
    Small MLP to process clinical metadata.

    Args:
        metadata_dim: Number of input metadata features.
        hidden_dim:   Hidden layer size (default 64).
        out_dim:      Output embedding size (default 32).
        dropout:      Dropout rate (default 0.3).
    """
    def __init__(self, metadata_dim: int, hidden_dim: int = 64,
                 out_dim: int = 32, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultimodalModel(nn.Module):
    """
    Late fusion multimodal model combining EfficientNet-B0 image features
    with clinical metadata features.

    Architecture:
        Image  → EfficientNet-B0 backbone → 1280-dim features
        Metadata → MLP → 32-dim features
        Fusion → Concat(1280+32) → 256 → NUM_CLASSES

    Args:
        metadata_dim:     Number of metadata input features.
        efficientnet_version: Backbone version (default 'b0').
        pretrained:       Load ImageNet weights (default True).
        dropout:          Dropout rate (default 0.3).
        metadata_hidden:  Hidden dim for metadata MLP (default 64).
        metadata_out:     Output dim for metadata MLP (default 32).
    """
    def __init__(
        self,
        metadata_dim: int,
        efficientnet_version: str = 'b0',
        pretrained: bool = True,
        dropout: float = 0.3,
        metadata_hidden: int = 64,
        metadata_out: int = 32,
    ):
        super().__init__()

        # ── Image branch ──────────────────────────────────────────────────────
        self.backbone = timm.create_model(
            f'efficientnet_{efficientnet_version}',
            pretrained=pretrained
        )
        self.image_out_dim = self.backbone.classifier.in_features
        # Remove original classifier
        self.backbone.classifier = nn.Identity()

        # ── Metadata branch ───────────────────────────────────────────────────
        self.metadata_branch = MetadataBranch(
            metadata_dim=metadata_dim,
            hidden_dim=metadata_hidden,
            out_dim=metadata_out,
            dropout=dropout,
        )

        # ── Fusion classifier ─────────────────────────────────────────────────
        fusion_dim = self.image_out_dim + metadata_out
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, image: torch.Tensor,
                metadata: torch.Tensor) -> torch.Tensor:
        # Image features
        img_features  = self.backbone(image)                # (B, 1280)

        # Metadata features
        meta_features = self.metadata_branch(metadata)      # (B, 32)

        # Late fusion
        fused = torch.cat([img_features, meta_features], dim=1)  # (B, 1312)

        return self.classifier(fused)                        # (B, NUM_CLASSES)

    def freeze_backbone(self):
        """Freeze backbone for Phase 1 training."""
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f'Frozen backbone — Trainable params: {trainable:,} / {total:,}')

    def unfreeze_all(self):
        """Unfreeze all layers for Phase 2 fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters())
        print(f'Unfrozen all layers — Trainable params: {trainable:,}')

    def get_image_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract image features before fusion (for XAI)."""
        return self.backbone(image)


if __name__ == "__main__":
    import torch

    for version in ["b0", "b1", "b2", "b3"]:
        print(f"\n── EfficientNet-{version.upper()} ──")
        info  = get_model_info(version)
        model = build_model(efficientnet_version=version, pretrained=False)

        # Test forward pass
        dummy = torch.randn(2, 3, info["input_size"], info["input_size"])
        out   = model(dummy)
        feats = get_features(model, dummy)

        print(f"  Input size:    {info['input_size']}x{info['input_size']}")
        print(f"  Params:        ~{info['params_M']}M")
        print(f"  Output shape:  {out.shape}")
        print(f"  Feature shape: {feats.shape}")