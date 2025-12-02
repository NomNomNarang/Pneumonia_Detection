<h1><b>HYBRID PNEUMONIA DETECTION MODEL</b></h1>

<p align="center"> <b>EfficientNetB0 + Classical Texture Features (GLCM)</b><br> A hybrid deep learning system for accurate Pneumonia detection from chest X-rays. </p> <p align="center"> <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge"> <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge"> <img src="https://img.shields.io/badge/Keras-DeepLearning-red?style=for-the-badge"> <img src="https://img.shields.io/badge/Model-Hybrid%20CNN%20%2B%20GLCM-green?style=for-the-badge"> <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"> </p>

📘 Overview

Pneumonia is a critical respiratory illness that requires fast and accurate diagnosis.
This project proposes a Hybrid Deep Learning architecture that combines:

EfficientNetB0 (pretrained CNN) → deep feature extraction
Handcrafted Classical Features (GLCM texture + statistical features) → local pattern analysis
Feature Fusion for stronger, more explainable predictions
The model classifies chest X-ray images into Normal or Pneumonia.

📂 Dataset

Kaggle: Chest X-Ray Pneumonia Dataset
🔗 https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

Split	NORMAL	PNEUMONIA	Total
Train	1341	3875	5216
Val	8	8	16
Test	234	390	624

Images are resized to 224×224, normalized, and preprocessed using EfficientNet standards.

🧠 Methodology
🔹 1. Preprocessing

Grayscale → RGB

Resize to 224×224

Normalization

EfficientNet preprocessing

🔹 2. Classical Feature Extraction (11 features)

Mean, Std, Min, Max

Entropy

GLCM Contrast

GLCM Energy

GLCM Homogeneity

GLCM Correlation

GLCM Dissimilarity

ASM

🔹 3. Deep Feature Extraction

EfficientNetB0 backbone

Frozen weights

GlobalAveragePooling2D

🔹 4. Feature Fusion

Deep + classical features combined using concatenation, followed by Dense layers and a Sigmoid classifier.

🧬 Model Architecture
Input Image → EfficientNetB0 → GAP → Deep Features
                                 ↓
                   Classical Texture Features (11D)
                                 ↓
                       Concatenate → Dense(128) → Dropout
                                 ↓
                              Sigmoid

📊 Evaluation

Key metrics observed during evaluation:

✔ Confusion Matrix
✔ ROC Curve (AUC)
✔ Accuracy, Precision, Recall, F1-score
✔ Training & Validation Curves
✔ Misclassified Image Visualizations

All plots are generated automatically in the notebook.

📈 Sample Visualization Outputs
<p align="center"> <img src="FIGURE_PLACEHOLDER_1" width="400"> <img src="FIGURE_PLACEHOLDER_2" width="400"> </p>
📁 Project Structure
.
├── hybrid_pneumonia_detection.ipynb
├── README.md
├── requirements.txt
├── saved_models/
└── figures/

⚙️ How to Run
1️⃣ Clone the repository
git clone https://github.com/YOUR_USERNAME/pneumonia-hybrid-model.git
cd pneumonia-hybrid-model

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Place dataset in Google Drive
MyDrive/chest_xray/train
MyDrive/chest_xray/val
MyDrive/chest_xray/test

🔮 Future Work
Add Grad-CAM explainability
Use deeper EfficientNet variants
Deploy as a Streamlit web app
Add clinical report generation

📝 License
This project is licensed under the MIT License.

🙌 Acknowledgments
Kaggle Dataset (Paul Mooney)
EfficientNet (Google Brain)
scikit-image
TensorFlow / Keras

4️⃣ Run the notebook
hybrid_pneumonia_detection.ipynb
