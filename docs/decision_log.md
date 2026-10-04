# Decision Log

## Decision 1: Google Colab for training
- Why: free GPU, same environment every time, reproducible
- Alternative rejected: training on laptop (too slow)
- Interview one-liner: "Colab gave me GPU access without hardware cost, and a consistent environment."

## Decision 2: PyTorch over Keras
- Why: industry/research standard for medical imaging, full control of training loop
- Alternative rejected: Keras (faster to write, hides internals)
- Interview one-liner: "Writing my own training loop gave me control over debugging and experiments."

## Decision 3: Code in Git, big files in Drive
- Why: Colab storage is temporary; Git is for small versioned files
- Interview one-liner: "I separated versioned code from large artifacts."

## Decision 4: Kaggle key in Colab Secrets
- Why: keys in notebooks leak when shared
- Interview one-liner: "I kept credentials out of source code."

## Decision 5: Fixed seed + central config
- Why: so results can be reproduced and compared fairly
- Interview one-liner: "Every run is reproducible from a saved config and seed."

## Decision 6: Raw data kept separate from processed data
- Why: original data is never modified, so any preprocessing can be redone
- Interview one-liner: "Raw data is immutable; everything downstream is regenerable."

## Decision 7: Frame the model as decision support, not diagnosis
- Why: it is the honest, safe framing, and how real clinical AI tools are positioned
- Alternative rejected: "AI diagnoses pneumonia" (overclaims, ignores regulation and risk)
- Interview one-liner: "I positioned it as triage support with a clinician in the loop."

## Decision 8: Recall (sensitivity) as the primary metric
- Why: a missed pneumonia case is far worse than a false alarm, and accuracy hides this with imbalanced classes
- Alternative rejected: accuracy (misleading: always guessing PNEUMONIA already scores about 70%)
- Interview one-liner: "I chose the metric based on the clinical cost of each error type."

## Decision 9: Scope the claim to the paediatric population in the data
- Why: the dataset is paediatric patients from one hospital, so claims beyond that are unsupported
- Alternative rejected: claiming general chest X-ray pneumonia detection
- Interview one-liner: "I stated the model's intended population and limits up front."

## Decision 10: Build a one-row-per-image metadata table before modeling
- Why: a single auditable source for labels, splits and patient IDs
- Interview one-liner: "I made the dataset inspectable before touching a model."

## Decision 11: Re-split data by patient, not by image
- Evidence: patients shared between the given splits = {'train-val': 0, 'train-test': 264, 'val-test': 0}
- Why: images from one patient must never appear in both train and test, or scores are inflated by leakage
- Interview one-liner: "I checked for patient-level leakage and split accordingly."

## Decision 12: Convert all images to 3-channel at load time
- Evidence: color modes found = {'L': 5573, 'RGB': 283}
- Why: pretrained models expect 3 channels, and mixed modes would break batching
- Interview one-liner: "I standardized input format to match the pretrained model."

## Decision 13: Checked for shortcut features before modeling
- Evidence: mean brightness by class = {'NORMAL': 122.1, 'PNEUMONIA': 124.3}
- Why: size/brightness differences between classes can let a model cheat
- Interview one-liner: "I tested whether trivial image properties could separate the classes."

## Decision 11 (evidence added): Size of the leakage
- 264 patients appear in both the original train and test folders (by class: {'PNEUMONIA': 170, 'NORMAL': 94})
- 416 of 624 original test images (66.7%) come from patients also present in train
- Both classes are affected, so this is not an artifact of a single filename pattern
- Assumption to state openly: patient ID is inferred from filenames
- Interview one-liner: "Two-thirds of the official test set came from patients seen in training, so I rebuilt the split at patient level."

## Decision 12 (evidence added): Grayscale conversion is lossless here
- 0 of 283 RGB images contain real color; they are grayscale stored as 3 channels

## Evidence note: aspect ratios
- Median 1.42, middle half 1.26 to 1.59, full range 0.84 to 3.38
- 27 images have aspect ratio above 2.5 (by class: {'PNEUMONIA': 27})
- Plan: keep a direct square resize for now. Squashing removes the raw size shortcut from the input; padding would expose it as black bars
- Possible later experiment: padding vs squashing, compared on the same split

## Decision 14: Re-split by patient, stratified on normal/bacteria/virus
- Evidence: 264 patients were shared between the original train and test folders
- Result: images per partition = {'train': 4177, 'val': 871, 'test': 808}; zero patients shared across partitions (asserted in code)
- Why: prevents leakage; stratifying on subtype keeps harder viral cases evenly spread
- Alternative rejected: using the provided splits (leaky, and class balance differed per split)
- Interview one-liner: "The original split leaked patients, so I rebuilt it at patient level and verified it with an assertion."

## Decision 15: Save the split to configs/splits.csv
- Why: every experiment uses identical data, and anyone can reproduce results
- Interview one-liner: "The split is a versioned file, not something regenerated on each run."

## Decision 16: PNEUMONIA is the positive class (label 1)
- Why: recall/sensitivity then means "fraction of pneumonia cases caught", matching the clinical goal

## Decision 17: Pre-resize to 256x256 grayscale PNG cache
- Why: original images are about 1300x970; reading them each epoch from Drive is slow. Resizing once is deterministic and much faster
- Alternative rejected: resizing on the fly every epoch

## Decision 18: Resize directly to 224x224 (no center crop)
- Why: lungs sit near the image edges, so cropping could cut off anatomy; a consistent squash is applied identically in train and test
- Alternative to test later: pad to square instead of squashing

## Decision 19: ImageNet mean/std normalization
- Why: Phase 5 uses ImageNet-pretrained weights, which expect inputs scaled this way

## Decision 20: Mild augmentation, no horizontal flip
- Why: small rotation/shift/zoom mimic positioning differences; brightness/contrast mimic exposure differences between machines. Flipping puts the heart on the wrong side, which is anatomically unrealistic
- Interview one-liner: "I chose augmentations that match real variation in X-ray acquisition and avoided ones that create impossible anatomy."

## Decision 20 (update): Switched augmentation interpolation to bilinear
- Evidence: visual check of augmented samples showed jagged, stair-stepped edges (RandomAffine defaults to nearest-neighbor)
- Why: interpolation artifacts are not real anatomy and could become learnable noise
- Interview one-liner: "I visually inspected augmented images before training and caught an interpolation artifact."

## Decision 13 (update): Shortcut check found real geometry and annotation differences
- Evidence: median width/height by class = {'NORMAL': {'width': 1654.0, 'height': 1323.0}, 'PNEUMONIA': {'width': 1160.0, 'height': 776.0}}
- Evidence: a classifier using ONLY width, height and aspect ratio reaches validation AUROC = 0.925
- Evidence: sample images show burned-in text ("A-P", "R" markers, technique labels) and different framing, more often on PNEUMONIA images (small sample, 4 per class)
- Why it matters: a model could score well by recognizing the acquisition source rather than lung disease
- Plan: (a) treat the geometry-only AUROC as the floor the CNN must clearly beat, (b) inspect Grad-CAM heatmaps, (c) run an occlusion test masking corners/text regions, (d) report results with this caveat
- Interview one-liner: "I found that image size alone separated the classes, so I measured how much a geometry-only model could cheat and used that as the baseline to beat."

## Decision 21: Baseline is a small CNN trained from scratch
- Why: simple, fast, and gives a score the pretrained model in Phase 5 must clearly beat
- Result (validation): AUROC 0.992, recall 0.949, specificity 0.978 at threshold 0.5
- Interview one-liner: "I built a simple baseline first so every later improvement could be measured."

## Decision 22: Plain BCE loss, no class weighting in the baseline
- Why: PNEUMONIA is the MAJORITY class (about 73%). Standard balancing would down-weight pneumonia, pushing the model toward missing it, the opposite of our clinical goal
- Instead: tune the decision threshold on validation to hit a recall target
- Later experiment: class-weighted loss as an ablation
- Interview one-liner: "Class imbalance ran in favor of the positive class, so I controlled recall with the threshold instead of reweighting."

## Decision 23: Select the best epoch by validation AUROC
- Why: AUROC does not depend on a threshold and is not fooled by class imbalance
- Alternative rejected: validation accuracy (always guessing pneumonia already scores about 73%)

## Decision 24: Sanity checks before training
- Initial loss near 0.69 and memorizing one batch catch bugs before long runs

## Decision 25: Test set not used in Phase 4
- Why: it is touched once, at the end, so the final number is honest
- Interview one-liner: "I selected models on validation only and kept the test set sealed until the final evaluation."
