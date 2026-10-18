
import numpy as np
import torch
import torch.nn.functional as F

H = 224

def region_masks(h=H):
    central = np.zeros((h, h), bool)
    central[int(.2 * h):int(.9 * h), int(.15 * h):int(.85 * h)] = True
    c = int(.25 * h)
    corners = np.zeros((h, h), bool)
    corners[:c, :c] = True
    corners[:c, -c:] = True
    b = int(.10 * h)
    border = np.ones((h, h), bool)
    border[b:-b, b:-b] = False
    return {"central": central, "periphery": ~central, "top_corners": corners, "outer_border": border}

def gradcam(net, x):
    """Grad-CAM for the pneumonia logit on the last DenseNet feature map."""
    net.eval()
    feats = net.net.features(x)
    act = F.relu(feats)
    logit = net.net.classifier(F.adaptive_avg_pool2d(act, 1).flatten(1)).squeeze(1)
    grads = torch.autograd.grad(logit.sum(), act)[0]
    w = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * act).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    cam = cam - cam.flatten(1).min(1)[0][:, None, None]
    cam = cam / (cam.flatten(1).max(1)[0][:, None, None] + 1e-8)
    return cam.detach().cpu().numpy(), torch.sigmoid(logit).detach().cpu().numpy()
