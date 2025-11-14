🌟 Hybrid Pneumonia Detection Model using EfficientNet + Classical Features

Deep Learning + Handcrafted Texture Features for Chest X-ray Classification


📚 Table of Contents

Overview

Dataset

Model Architecture

Features Used

Setup & Installation

How to Use

Training

Evaluation

Visualizations

Results

Folder Structure

Future Work

License

Citation

🧠 Overview

This project presents a hybrid deep learning model for detecting Pneumonia from chest X-ray images.
Unlike traditional CNN-only approaches, this model combines:

✔ Deep features

Extracted using EfficientNet-B0, pretrained on ImageNet.

✔ Classical features

Based on GLCM texture, entropy, and intensity statistics.

These two feature sets are concatenated to create a more expressive representation of lung structure abnormalities.

Hybrid models often perform better in medical imaging because handcrafted texture features capture local patterns that CNNs may ignore.

📂 Dataset

We use the Kaggle public dataset:

📌 Chest X-Ray Images (Pneumonia)
🔗 https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

Dataset Split
Split	NORMAL	PNEUMONIA	Total
Train	1341	3875	5216
Val	8	8	16
Test	234	390	624
Folder Structure (Expected)
chest_xray/
│── train/
│    ├── NORMAL
│    └── PNEUMONIA
│── val/
│    ├── NORMAL
│    └── PNEUMONIA
│── test/
     ├── NORMAL
     └── PNEUMONIA

🏗 Model Architecture
Input Image → EfficientNetB0 (Frozen Layers)
                        ↓
          GlobalAveragePooling2D
                        ↓
               Deep Feature Vector
                        ↓
Handcrafted Texture Features (11D vector)
                        ↓
             Concatenation Layer
                        ↓
                Dense (128) + Dropout
                        ↓
                 Sigmoid Output


A complete diagram can be added if needed — ask: “Generate architecture diagram”.

🔍 Features Used
Deep Features

✔ EfficientNet-B0 feature maps
✔ Transfer learning (no retraining base layers)

Classical Features (11 total)

Mean pixel value

Standard deviation

Min / Max

Entropy

GLCM:

Contrast

Energy

Homogeneity

Correlation

Dissimilarity

ASM

These texture features are commonly used in radiographic analysis.

⚙️ Setup & Installation

Clone the repo:

git clone https://github.com/YOUR_USERNAME/chest-xray-hybrid-model.git
cd chest-xray-hybrid-model


Install required packages:

pip install -r requirements.txt


Requirements include:

TensorFlow

scikit-learn

scikit-image

numpy

matplotlib

seaborn

tqdm

🧪 How to Use

Place the dataset inside Google Drive:

MyDrive/chest_xray/train
MyDrive/chest_xray/val
MyDrive/chest_xray/test


Open the notebook:

hybrid_pneumonia_detection.ipynb


Run all cells sequentially.

🏋️ Training

The model trains using:

Adam optimizer

Binary cross-entropy loss

Class weights (balanced)

Batch size: 32

Image size: 224×224

Epochs: 10–20 recommended

📊 Evaluation

Metrics computed:

Accuracy

Precision

Recall

F1-Score

AUC (Area Under ROC)

Confusion Matrix

Predictions:

preds = model.predict(test_seq).ravel()

🎨 Visualizations

The code automatically generates:

✔ Confusion Matrix Heatmap
✔ ROC Curve
✔ Training Accuracy Curve
✔ Training Loss Curve
✔ Histogram of Predictions
✔ Misclassified Image Examples

Example:

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")


These figures help in discussing performance in the research paper.

🏆 Results

(Replace with your actual numbers after training.)

Accuracy: XX.X%

Precision: XX.X%

Recall: XX.X%

F1 Score: XX.X%

AUC: 0.XXX

The hybrid model typically improves robustness compared to pure CNNs.

📁 Folder Structure
.
├── hybrid_pneumonia_detection.ipynb
├── README.md
├── requirements.txt
├── saved_models/
├── figures/
└── utils/

🔮 Future Work

Add Grad-CAM visualization

Use EfficientNetB3/B4 for deeper features

Implement attention-based fusion

Add explainable AI (XAI) module

Deploy model as a web or mobile app

📜 License

This project is licensed under the MIT License.

📖 Citation

If you use this project, please cite:

@article{your2025pneumonia,
  title={Hybrid CNN + Classical Texture Features for Pneumonia Detection from Chest X-rays},
  author={Your Name},
  year={2025},
  journal={GitHub Repository},
  url={https://github.com/YOUR_USERNAME/chest-xray-hybrid-model}
}
