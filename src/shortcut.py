
import numpy as np
from sklearn.metrics import roc_auc_score

def fit_bin_edges(df, n_area=4, n_aspect=3):
    """Bin edges from image dimensions only (no labels involved)."""
    area = np.log(df.width.to_numpy() * df.height.to_numpy())
    aspect = df.width.to_numpy() / df.height.to_numpy()
    return {
        "area": np.quantile(area, np.linspace(0, 1, n_area + 1)[1:-1]),
        "aspect": np.quantile(aspect, np.linspace(0, 1, n_aspect + 1)[1:-1]),
    }

def assign_bins(df, edges):
    a = np.digitize(np.log(df.width.to_numpy() * df.height.to_numpy()), edges["area"])
    r = np.digitize(df.width.to_numpy() / df.height.to_numpy(), edges["aspect"])
    return a * (len(edges["aspect"]) + 1) + r

def compute_weights(df, bin_ids, min_per_class=3, clip=10.0):
    """Weight = P(class) / P(class | bin). Inside each bin, both classes then carry the same
    share as in the whole set, so size and shape no longer predict the class."""
    y = df.target.to_numpy()
    w = np.zeros(len(df))
    p_y = {c: (y == c).mean() for c in (0, 1)}
    for b in np.unique(bin_ids):
        m = bin_ids == b
        n_b = m.sum()
        counts = {c: int(((y == c) & m).sum()) for c in (0, 1)}
        if min(counts.values()) < min_per_class:
            continue
        for c in (0, 1):
            w[m & (y == c)] = p_y[c] * n_b / counts[c]
    w = np.minimum(w, clip)
    kept = w > 0
    w[kept] = w[kept] / w[kept].mean()
    return w

def weighted_metrics(probs, ys, w, thr=0.5):
    keep = w > 0
    p, y, ww = probs[keep], ys[keep], w[keep]
    pred = (p >= thr).astype(int)
    tp = ww[(pred == 1) & (y == 1)].sum(); fn = ww[(pred == 0) & (y == 1)].sum()
    tn = ww[(pred == 0) & (y == 0)].sum(); fp = ww[(pred == 1) & (y == 0)].sum()
    return {
        "auroc": float(roc_auc_score(y, p, sample_weight=ww)),
        "recall": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "images_used": int(keep.sum()),
    }
