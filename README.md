# BreastCancer-Mass-Classification
Capstone: Mammographic mass malignancy prediction using ML &amp; DL (UCI dataset)
# Breast Cancer Mass Classification (Capstone)

Predict benign vs. malignant mammographic masses using **Machine Learning** and **Deep Learning**.

## 🚀 Project Overview
- **Dataset**: [Mammographic Masses (UCI ML Repository)](https://archive.ics.uci.edu/ml/datasets/Mammographic+Mass)  
- **Goal**: Reduce false positives in mammogram results with supervised ML.  
- **Features**: Age, Shape, Margin, Density.  
- **Target**: Severity (benign=0, malignant=1).  
- **Missing values**: handled with median imputation.  

## 🧠 Models
1. **Deep Learning (MLP)**  
   - 2 hidden layers (16 → 8), ReLU, Adam.  
   - Results: **Accuracy ~80.5%**, **ROC-AUC ~0.889**.  

2. **Model Comparison (Classical ML)**  
   - Logistic Regression, Decision Tree, Random Forest, SVM (RBF).  
   - Best: **SVM (Accuracy ~80.1%, ROC-AUC ~0.873)**.  

## 📊 Evaluation
- Accuracy, Precision, Recall, F1-score, ROC-AUC.  
- Confusion Matrices and ROC curves saved in `outputs/`.

## ⚙️ Usage
```bash
pip install -r requirements.txt

# Run Neural Network
python main.py --mlp

# Run Model Comparison
python main.py --compare
