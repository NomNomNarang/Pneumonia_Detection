⭐ Hybrid CNN + Classical Features for Pneumonia Detection from Chest X-ray Images
(EfficientNetB0 + GLCM Texture Features)

This project implements a hybrid deep learning model that combines:

EfficientNetB0 (transfer learning) for deep visual features

Classical handcrafted texture features (GLCM, entropy, intensity stats)

to classify Normal vs Pneumonia chest X-ray images.

The model is built, trained, evaluated, and visualized using the Kaggle Chest X-ray Pneumonia dataset.

🚀 Key Features
✔ Hybrid Model Architecture

EfficientNetB0 (pretrained on ImageNet)

11 handcrafted features:

Mean, Std, Min, Max

Entropy

GLCM Contrast

Homogeneity

Energy

Correlation

Dissimilarity

ASM

Feature fusion using concatenation

Dense layers on top for classification

📊 Dataset

Kaggle: Chest X-Ray Images (Pneumonia)
Link: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

Split	NORMAL	PNEUMONIA	Total
Train	1341	3875	5216
Val	8	8	16
Test	234	390	624
🔧 Pipeline Overview
1. Preprocessing

Convert to grayscale

Resize to 224×224

Normalize intensities

EfficientNet preprocessing

2. Feature Extraction

Classical features computed using GLCM + statistics

Deep features extracted from EfficientNetB0 backbone

Both feature sets are concatenated

3. Training

Class-balanced training

Adam optimizer

Binary cross-entropy loss

Callback visualizations

4. Evaluation

Classification report

Accuracy, Precision, Recall, F1

AUC and ROC Curve

Confusion Matrix Heatmap

Prediction distribution

Misclassified image visualization

📈 Visualization Outputs

The notebook generates:

Confusion matrix heatmap

ROC curve with AUC

Training accuracy & loss curves

Prediction histogram

Misclassified image samples

All figures are saved automatically (optional).

🧪 Results Summary

(Add your actual numbers)

Accuracy: xx.x%

Precision: xx.x%

Recall: xx.x%

F1 Score: xx.x%

AUC: 0.xxx

🗂 Project Structure
├── Features Extraction
├── EfficientNet Hybrid Model
├── Data Preprocessing
├── Model Training
├── Evaluation & Plots
└── Saved Model / Figures (optional)

🧰 Dependencies

Python 3

TensorFlow / Keras

Scikit-learn

Scikit-image

NumPy

Matplotlib / Seaborn

tqdm

📌 How to Run

Mount your Google Drive

Place dataset in:
MyDrive/chest_xray/train, val, test

Run the notebook sequentially

Evaluation and plots will be generated automatically

🙌 Acknowledgments

Dataset provided by:
Paul Mooney – Chest X-Ray Pneumonia Kaggle Dataset

EfficientNet:
Tan & Le (Google Brain), 2019
