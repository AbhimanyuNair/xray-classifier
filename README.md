# Chest X-ray Pneumonia Classifier

A DenseNet121 image-classification project exploring pneumonia detection from chest X-rays. The dashboard presents model results, evaluation metrics, and Grad-CAM visualizations, with a focus on making model behavior easier to inspect.

> **Demo:** [Open the live dashboard](https://xray-classifier-wcxoeqqxysfefmxyuw2wji.streamlit.app/)

## Dashboard

The hosted dashboard runs in lite mode. It includes precomputed predictions and Grad-CAM heatmaps for selected test images, along with interactive charts and threshold controls. It does not load the trained model weights, so uploading a new X-ray for live inference is not available in the hosted version.

For live inference on a new image, use the Colab notebook: [Open the Colab notebook](YOUR_COLAB_NOTEBOOK_URL).

## Models

The project compares two DenseNet121 classifiers:

- **Full-image model:** uses the full resized X-ray.
- **Center-only model:** masks the image periphery and uses the central chest region.

A geometry-only baseline is also included in the dashboard for comparison. It uses image dimensions rather than image pixels.

## Evaluation and explainability

The dashboard provides test-set metrics, prediction exploration, threshold controls, and Grad-CAM heatmaps for selected examples. The Colab workflow reproduced the Phase 7 sample probabilities, with a maximum difference of `0.0` in the Phase 8.9a check.

Grad-CAM visualizations indicate image regions that influenced a model prediction. They are a model explanation aid, not a clinical finding.

## Run the dashboard locally

Clone the repository and install the hosted dashboard dependencies:

```bash
git clone https://github.com/AbhimanyuNair/xray-classifier/tree/main
cd xray-classifier
pip install -r app/requirements.txt
streamlit run app/app.py
