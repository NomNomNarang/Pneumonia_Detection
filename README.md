**🫁 Pneumonia Detection using Hybrid Deep Learning (EfficientNet + Classical Features)
**

📘 Overview

Pneumonia is a severe lung infection that can be life-threatening if not diagnosed early.
Chest X-rays are the most widely used tool for detection —but manual interpretation is time-consuming and prone to errors.

This project presents a Hybrid Deep Learning Model combining:

EfficientNetB0 (pretrained CNN) for extracting deep visual features

Handcrafted Classical Features (GLCM texture + intensity features) for enhancing interpretability

Feature Fusion to improve pneumonia detection performance

The goal is to build a robust and explainable diagnostic tool for binary classification:
Normal vs Pneumonia.

📊 Dataset

This project uses the well-known Chest X-Ray Pneumonia Dataset from Kaggle.

🔗 Dataset Link:
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

Dataset Distribution
Split	NORMAL	PNEUMONIA	Total
Train	1341	3875	5216
Val	8	8	16
Test	234	390	624

The dataset shows significant class imbalance, which is handled using class weights.

🧠 Methodology
🔹 1. Data Preprocessing

Image resizing to 224×224

Grayscale → RGB conversion

Normalization

EfficientNet preprocessing

🔹 2. Classical Feature Extraction

Using scikit-image, the following 11 features are extracted:

Mean, Standard Deviation, Min, Max

Entropy

GLCM Contrast

GLCM Energy

GLCM Homogeneity

GLCM Correlation

GLCM Dissimilarity

GLCM ASM

🔹 3. Deep Feature Extraction

EfficientNetB0

Pretrained on ImageNet

Feature map → GlobalAveragePooling2D

🔹 4. Hybrid Feature Fusion

Deep + Classical features are concatenated and passed through:

Dense(128)

Dropout(0.3)

Output layer with Sigmoid activation

🧬 Model Architecture
Image Input → EfficientNetB0 → GAP → Deep Features
                                 ↓
                   Classical Texture Features
                                 ↓
                       Concatenate → Dense → Output

🏋️ Training

Optimizer: Adam (1e-4)

Loss: Binary Cross Entropy

Metrics: Accuracy, AUC

Class balancing using compute_class_weight()

Batch size: 32

Epochs: 10–20

📈 Evaluation Metrics

The model is evaluated using:

✔ Accuracy

✔ Precision

✔ Recall

✔ F1-score

✔ Confusion Matrix

✔ ROC Curve + AUC

✔ Prediction Distribution

All visualizations are included in the notebook.

📉 Visualizations

This project provides:

🔹 Confusion Matrix
🔹 ROC Curve
🔹 Training Accuracy/Loss Curves
🔹 Histogram of Prediction Probabilities
🔹 Misclassified Sample Images

These help analyze model performance and error patterns.

📁 Project Structure
├── hybrid_pneumonia_detection.ipynb
├── README.md
├── requirements.txt
├── utils/
├── saved_models/
└── figures/

🚀 How to Run

Download the Kaggle dataset

Place it in Google Drive:

MyDrive/chest_xray/
    train/
    val/
    test/


Open the notebook in Google Colab

Run all cells sequentially

View evaluation metrics and generated plots

🧾 Results (Add your values here)

Accuracy: XX.X%

Precision: XX.X%

Recall: XX.X%

F1 Score: XX.X%

AUC: 0.XXX

📜 License

This project is licensed under the MIT License.

🙌 Acknowledgments

Dataset by Paul Mooney, Kaggle

EfficientNet by Tan & Le (Google Brain)

scikit-image for classical feature extraction
