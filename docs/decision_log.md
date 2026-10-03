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
