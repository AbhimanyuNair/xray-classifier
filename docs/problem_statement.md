# Problem Statement

## Clinical problem
Pneumonia is a leading cause of illness in young children, and diagnosis from a
chest X-ray depends on radiologist availability. Delays slow treatment.

## ML task
Binary image classification: chest X-ray -> NORMAL or PNEUMONIA.

## Intended use (decision support, NOT diagnosis)
Flag likely-pneumonia X-rays so a radiologist can review them sooner.
A clinician always makes the final call.

## Intended population
Paediatric patients (ages ~1-5), matching the training dataset.
The model must not be assumed valid for adults or other hospitals.

## Success metrics
- Primary: recall (sensitivity) for PNEUMONIA - missing a sick patient is the costly error
- Secondary: specificity, precision, AUROC, confusion matrix
- Report results on a patient-level held-out test set, never on training data

## Known risks and limitations
- Single-source dataset (one hospital): may not generalize to other scanners/sites
- Class imbalance (about 3x more pneumonia than normal)
- Original validation set is only 16 images: we will re-split the data ourselves
- Possible shortcut learning (model reading image artefacts instead of lungs):
  we will check this with Grad-CAM heatmaps
