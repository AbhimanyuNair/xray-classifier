
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        probs.append(torch.sigmoid(model(x.to(device))).cpu())
        ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()

def metrics_at(probs, ys, thr=0.5):
    pred = (probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(ys, pred, labels=[0, 1]).ravel()
    return {
        "auroc": float(roc_auc_score(ys, probs)),
        "recall": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.float().to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n

def fit(model, loaders, criterion, optimizer, device, epochs, patience, ckpt_path, metric_fn=None):
    """Train, pick the best epoch by validation AUROC, stop early if it stalls."""
    history, best, bad = [], -1.0, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        probs, ys = predict(model, loaders["val"], device)
        val_loss = float(F.binary_cross_entropy(torch.tensor(probs), torch.tensor(ys).float()))
        m = metric_fn(probs, ys) if metric_fn else metrics_at(probs, ys)
        history.append({"epoch": ep, "train_loss": tr_loss, "val_loss": val_loss,
                        "val_auroc": m["auroc"], "val_recall": m["recall"],
                        "val_specificity": m["specificity"]})
        flag = ""
        if m["auroc"] > best:
            best, bad = m["auroc"], 0
            torch.save(model.state_dict(), ckpt_path)
            flag = "  <- best"
        else:
            bad += 1
        print(f"ep {ep:02d} | train {tr_loss:.3f} | val {val_loss:.3f} | AUROC {m['auroc']:.3f} "
              f"| recall {m['recall']:.3f} | spec {m['specificity']:.3f} | {time.time()-t0:.0f}s{flag}")
        if bad >= patience:
            print(f"Early stop: no AUROC improvement for {patience} epochs.")
            break
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return pd.DataFrame(history)
