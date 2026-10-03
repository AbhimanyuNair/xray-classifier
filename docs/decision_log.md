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
