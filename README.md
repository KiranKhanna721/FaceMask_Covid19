# Face Mask and COVID-19 Detection

A Flask web app with two deep learning image classifiers:

- **Face mask detection:** finds whether a person in an uploaded photo is wearing a mask (`model-facemask.h5`)
- **COVID-19 detection:** classifies chest X-ray images as COVID positive or negative (`covid.h5`), with a confidence percentage

## Tech stack

Python, Flask, TensorFlow / Keras, OpenCV, NumPy, plus HTML/CSS/Bootstrap for the front end

## Project structure

| Path | Purpose |
|------|---------|
| `app.py` | Flask server: image upload, preprocessing (224×224) and prediction |
| `Covid.ipynb` | Training notebook for the COVID-19 X-ray model |
| `covid.h5` | Trained COVID-19 model |
| `templates/`, `static/` | Web pages and assets |

## Getting started

```bash
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000.

> `model-facemask.h5` isn't in the repo. Add it before running the mask detector. This is not a medical diagnostic tool.
