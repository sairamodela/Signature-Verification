# Signature Verification System Using Machine Learning

A signature verification system that classifies signatures as Genuine or Forged using a Random Forest classifier with handcrafted image features.

---

## Project Overview

| Item | Detail |
|---|---|
| **Algorithm** | Random Forest Classifier |
| **Dataset** | CEDAR Signature Dataset |
| **Features** | HOG, LBP, Geometric, Projection |
| **Interface** | Flask Web Application |
| **Language** | Python 3.10+ |

---

## Project Structure

```
signature-verification/
├── app.py                  # Flask web application
├── requirements.txt
├── src/
│   ├── features.py         # Feature extraction (HOG, LBP, Geometric, Projection)
│   ├── train.py            # Model training pipeline
│   └── predict.py          # Prediction module
├── models/
│   ├── signature_model.pkl # Trained Random Forest model
│   └── scaler.pkl          # StandardScaler
├── templates/
│   └── index.html          # Web UI
├── static/
│   └── uploads/            # Uploaded images (auto-created)
├── notebooks/
│   └── signature_verification.ipynb   # EDA + training walkthrough
└── data/
    ├── genuine/            # Genuine signature images
    └── forged/             # Forged signature images
```

---

## How It Works

### Feature Extraction
The system extracts 4 types of features from each signature image:

| Feature Type | Description | Size |
|---|---|---|
| **HOG** | Histogram of Oriented Gradients — edge & gradient patterns | 1764 |
| **LBP** | Local Binary Pattern — local texture | 10 |
| **Geometric** | Pixel density, centroid, bounding box, aspect ratio | 6 |
| **Projection** | Horizontal & vertical ink distribution | 64 |

**Total feature vector: 1844 dimensions**

### Model
- **Random Forest** with 200 trees
- **StandardScaler** normalisation
- **80/20 train-test split** with stratification
- **5-fold cross-validation** for robust evaluation

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/sairamodela/signature-verification.git
cd signature-verification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare dataset
Download the [CEDAR Signature Dataset](http://www.cedar.buffalo.edu/NIJ/data/) and place images in:
```
data/genuine/   ← genuine signature images
data/forged/    ← forged signature images
```

### 4. Train the model
```bash
python src/train.py
```

### 5. Run the web app
```bash
python app.py
```
Open your browser at `http://localhost:5000`

---

## Results

| Metric | Score |
|---|---|
| **Accuracy** | ~96% |
| **Precision (Genuine)** | ~97% |
| **Recall (Genuine)** | ~95% |
| **5-Fold CV Accuracy** | ~95.5% |

> Results may vary depending on dataset size and preprocessing.

---

## Web Application

Upload any signature image (PNG/JPG) and get:
- Genuine or Forged prediction
- Confidence score
- Probability bars for each class

---

## Tech Stack

- **Python** — Core language
- **scikit-learn** — Random Forest, preprocessing, metrics
- **OpenCV** — Image processing
- **scikit-image** — HOG and LBP feature extraction
- **Flask** — Web framework
- **Bootstrap 5** — Frontend UI

---

## Author

**Sairam Odela**  
[LinkedIn](https://www.linkedin.com/in/sairam-odela-801462250/) · [Portfolio](https://sairamodela.github.io/sai-portfolio/)
