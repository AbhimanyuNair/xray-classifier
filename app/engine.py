"""Self-contained inference helpers for the dashboard.

Everything here mirrors the training pipeline from the project notebooks:
  - cache step:  grayscale -> 256x256 (Lanczos)
  - eval step:   resize 224x224 -> tensor -> ImageNet normalisation
  - model B:     periphery blacked out so only the central chest is visible
  - Grad-CAM:    last DenseNet feature map, pneumonia logit
torch is imported lazily so the dashboard can still run in "precomputed" mode
if torch or the weights are unavailable.
"""
from pathlib import Path

import numpy as np
from PIL import Image

H = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def region_masks(h=H):
    """Fixed regions (same definition as the explainability phase)."""
    central = np.zeros((h, h), bool)
    central[int(.2 * h):int(.9 * h), int(.15 * h):int(.85 * h)] = True
    return {"central": central, "periphery": ~central}


def torch_available():
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        return True
    except Exception:
        return False


def build_model():
    import torch.nn as nn
    from torchvision import models as tvm

    class DenseNetClassifier(nn.Module):
        def __init__(self, dropout=0.2):
            super().__init__()
            self.net = tvm.densenet121(weights=None)
            self.net.classifier = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(self.net.classifier.in_features, 1)
            )

        def forward(self, x):
            return self.net(x).squeeze(1)

    return DenseNetClassifier()


def load_model(path):
    import torch

    model = build_model()
    state = torch.load(str(path), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def to_gray256(pil_img):
    """Same as the training cache: grayscale, 256x256, Lanczos."""
    return pil_img.convert("L").resize((256, 256), Image.Resampling.LANCZOS)


def preprocess(gray256, center_only):
    """Return the model input tensor (1,3,224,224)."""
    import torch
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize((H, H)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    x = tf(gray256.convert("RGB")).unsqueeze(0)
    if center_only:
        mask = torch.tensor(region_masks()["periphery"])[None, None]
        black = ((0 - torch.tensor(MEAN)) / torch.tensor(STD))[None, :, None, None]
        x = torch.where(mask, black.expand_as(x), x)
    return x


def predict_with_cam(model, x):
    """Return (probability of pneumonia, Grad-CAM map in [0,1] of shape 224x224)."""
    import torch
    import torch.nn.functional as F

    model.eval()
    feats = model.net.features(x)
    act = F.relu(feats)
    logit = model.net.classifier(F.adaptive_avg_pool2d(act, 1).flatten(1)).squeeze(1)
    grads = torch.autograd.grad(logit.sum(), act)[0]
    w = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * act).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    cam = cam - cam.flatten(1).min(1)[0][:, None, None]
    cam = cam / (cam.flatten(1).max(1)[0][:, None, None] + 1e-8)
    prob = float(torch.sigmoid(logit).detach().cpu().numpy()[0])
    return prob, cam.detach().cpu().numpy()[0]


def masked_preview(gray256, center_only):
    """What the model actually sees, as a displayable 224x224 image."""
    g = np.asarray(gray256.resize((H, H), Image.Resampling.BILINEAR)).copy()
    if center_only:
        g[region_masks()["periphery"]] = 0
    return Image.fromarray(g).convert("RGB")


def overlay(gray256, cam, alpha=0.45):
    """Blend a jet-coloured Grad-CAM onto the grayscale image."""
    import matplotlib

    base = np.asarray(gray256.resize((H, H), Image.Resampling.BILINEAR).convert("RGB")).astype(float) / 255.0
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    out = (1 - alpha) * base + alpha * heat
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))
