"""
ml_unet_model.py — ReefUNet: arquitetura canónica para segmentação de recifes.

Implementação pura PyTorch — sem segmentation-models-pytorch.
CPU-only (DGT JupyterHub AMD EPYC-Milan 96 cores, sem GPU).

Arquitetura canónica (ver reef_pipeline_system_prompt_v2.md):
  - Encoder: 4 níveis, features=[64, 128, 256, 512]
  - Bottleneck: 1024 canais
  - Decoder: ConvTranspose2d + DoubleConv em cada nível
  - Output: sigmoid [0, 1] aplicado internamente em forward()
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ── Bloco base ────────────────────────────────────────────────────────────────


class DoubleConv(nn.Module):
    """Dois Conv2d → BatchNorm → ReLU consecutivos."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Arquitectura canónica ─────────────────────────────────────────────────────


class ReefUNet(nn.Module):
    """
    U-Net para segmentação de recifes em MDT rasters.

    Input:  (B, 1, H, W)  — MDT normalizado [0, 1] (qualquer múltiplo de 16)
    Output: (B, 1, H, W)  — máscara de probabilidade sigmoid [0, 1]

    Arquitectura canónica:
        Encoder: 4 níveis com features=[64, 128, 256, 512]
        Bottleneck: 1024 canais
        Decoder: ConvTranspose2d (aprende o upsample) + DoubleConv + skip
        Final: Conv1×1 → sigmoid

    31 M parâmetros. Recomendado: tiles 256×256 em inferência.
    Treino em CPU com batch_size=8 numa máquina de 96 cores ≈ 80 s/epoch.

    NUNCA chamar .cuda() ou .to('cuda') — esta máquina não tem GPU.
    NUNCA importar segmentation-models-pytorch — não está instalado no HPC.
    """

    def __init__(self, in_channels: int = 1, features: list = None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
            self.ups.append(DoubleConv(f * 2, f))
        self.final = nn.Conv2d(features[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[-(i // 2 + 1)]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)
        return torch.sigmoid(self.final(x))

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_trained_model(checkpoint_path: str, device: str = "cpu") -> ReefUNet:
    """Carrega ReefUNet de um ficheiro de pesos e coloca em modo eval."""
    model = ReefUNet().to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    log.info("ReefUNet carregado: %s  (%d params)", checkpoint_path, model.num_parameters)
    return model


# ── Loss & Métricas ───────────────────────────────────────────────────────────


class _ReefLoss(nn.Module):
    """
    BCE ponderado + Soft Dice para segmentação com desequilíbrio de classes.

    Recebe probabilidades sigmoid [0, 1] — o modelo já aplica sigmoid em forward().
    pos_weight: peso dos pixels positivos (recife) vs negativos (fundo).
    Dataset: ≈ 94% patches vazios → usar pos_weight ≈ 10 para não colapsar.
    """

    def __init__(self, pos_weight: float = 10.0):
        super().__init__()
        self._pw = pos_weight

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE com peso diferenciado: pixels de recife valem pos_weight vezes mais
        w = targets * (self._pw - 1.0) + 1.0
        bce = F.binary_cross_entropy(probs, targets, weight=w)
        # Soft Dice: penaliza desalinhamento geométrico em patches pequenos
        inter = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = 1.0 - (2.0 * inter + 1e-6) / (union + 1e-6)
        return bce + dice.mean()


def get_loss_function(pos_weight: float = 10.0) -> _ReefLoss:
    """
    Retorna BCE + Dice combinados para outputs sigmoid [0, 1].

    pos_weight compensa a sub-representação de pixels de recife (≈ 6% do dataset).
    Valor recomendado: 10.0 para 94% de patches vazios.
    """
    return _ReefLoss(pos_weight=pos_weight)


def calculate_iou(probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Binary IoU (Jaccard) para saídas sigmoid do ReefUNet.

    Args:
        probs:     probabilidades sigmoid [0, 1] — saída direta do modelo
        targets:   máscara binária ground-truth  (B, 1, H, W)
        threshold: limiar de decisão (padrão 0.5)

    Returns:
        IoU médio do batch como float em [0, 1].
        Retorna 1.0 se union==0 (ambos vazios → concordância perfeita).
    """
    preds = (probs > threshold).float()
    inter = (preds * targets).sum()
    union = preds.sum() + targets.sum() - inter
    if union == 0:
        return 1.0
    return (inter / union).item()
