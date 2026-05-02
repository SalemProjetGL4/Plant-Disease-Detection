# Plant Disease Detection

Streamlit app for plant disease classification with Grad-CAM visualization.

## Features
- Single-image inference with prediction confidence
- Grad-CAM heatmap toggle
- Class label mapping via classes.json
- Consistent preprocessing with the evaluation notebook

## Project structure
- app/: Streamlit app and inference code
- datasets/: raw and preprocessed datasets
- models/: trained model checkpoints (.pth)
- notebooks/: training and evaluation notebooks

## Dataset layout
Put the dataset in this folder like this:

```
datasets/
  New Plant Diseases/
    train/
    valid/
  test/
```

If you use preprocessed data, keep the same layout under datasets/preprocessed/.

## Setup
1) Create a Python environment
2) Install dependencies

```
pip install -r requirements.txt
```

## Run the app
From the project root:

```
streamlit run app/app.py
```

## Models
The app defaults to:

```
models/2 stage fine tuned model.pth
```

If you want to use a different checkpoint, update DEFAULT_MODEL_PATH in app/app.py.

## Class labels
Class labels are loaded from classes.json (mapping raw class name -> display label).
If classes.json is missing, the app builds labels from the dataset folder names.

Example classes.json:

```
{
  "Tomato___healthy": "Tomato - healthy",
  "Tomato___Early_blight": "Tomato - Early blight"
}
```

## Notes on preprocessing
The app uses the same Resize + Normalize preprocessing as the evaluation notebook.
If you want the stronger preprocessing (median blur + CLAHE + sharpen), switch
preprocessing in app/inference.py.

## Troubleshooting
- If predictions look random, verify the model checkpoint path exists.
- If labels look wrong, verify classes.json matches the dataset class order.
- If running from a subfolder, keep classes.json at the project root.
