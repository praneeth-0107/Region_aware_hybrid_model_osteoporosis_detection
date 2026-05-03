# 🦴 A Region-Aware Hybrid Framework for Osteoporosis Detection
### Femur & Tibia-Based Deep Learning System with Ensemble Learning

---

## 📌 Overview
This project presents a **region-aware hybrid deep learning framework** for detecting **osteoporosis, osteopenia, and normal bone conditions** from knee X-ray images.

Unlike traditional approaches that analyze the entire knee, this system:
- Separates **Femur and Tibia**
- Performs **independent classification**
- Combines predictions using an **AUC-weighted ensemble**

---

## 🚀 Key Features
- Automatic **Femur & Tibia separation**
- Multi-model training (6 architectures)
- Best model selection using Accuracy & AUC
- **AUC-weighted ensemble learning**
- Bone-specific prediction with confidence %

---

## 🧠 Pipeline
```

Input Knee X-ray
↓
Preprocessing & ROI Extraction
↓
Bone Segmentation
↓
Femur & Tibia Separation
↓
Model Training (6 Architectures)
↓
Best Model Selection
↓
AUC-Weighted Ensemble
↓
Final Prediction (Severity + Confidence %)

```

---

## 📊 Dataset

### 🔹 Original Dataset
https://www.kaggle.com/code/mohamedgobara/knee-osteoporosis-multiclass-xception-89-2/input

### 🔹 Segmented Dataset
https://drive.google.com/file/d/177qqHeZfWQ1GKSLmcwWnZjDkwM58FMfW/view?usp=sharing

### 🔹 Dataset Details
- Classes:
  - Osteoporosis
  - Osteopenia
  - Normal

- Data Split:
  - Train: **80%**
  - Validation: **10%**
  - Test: **10% (held-out)**

---

## 🤖 Models Used
Each model is trained separately on **Femur and Tibia datasets**:

- ResNet50  
- DenseNet121  
- VGG16  
- EfficientNet-B3  
- EfficientNet-V2-S ⭐ (Best)  
- Swin Transformer  

---

## 🏆 Best Model Selection
- Models evaluated using:
  - Accuracy
  - Balanced Accuracy
  - ROC-AUC

- Best models selected:
  - **Best Femur Model:** EfficientNet-V2-S  
  - **Best Tibia Model:** EfficientNet-V2-S  

---

## 🔗 Ensemble Method
Final prediction is computed using **AUC-weighted averaging**:





<img width="420" height="105" alt="image" src="https://github.com/user-attachments/assets/c9b7e534-fb30-4a48-bb44-8b48d4f3bcb8" />




### Advantages:
- Improves accuracy
- Reduces bias
- Combines strengths of both models

---

## 📈 Results
- Femur Accuracy: **~85.6%**  
- Tibia Accuracy: **~82.5%**  
- Ensemble Accuracy: **~84.6%**  
- Macro AUC: **~0.9579**

---

## 🧾 Output Example


Diagnosis: Osteoporosis
Risk Level: High

Femur:
Confidence: 95.8%

Tibia:
Confidence: 96.4%

Final Ensemble Confidence: 96.1%



---

## 📂 Project Structure


├── models/
│   ├── efficientnet_v2_s_femur.pth
│   ├── efficientnet_v2_s_tibia.pth
│   ├── resnet50_femur.pth
│   └── ...
│
├── results/
│   ├── reports
│   ├── confusion matrices
│   ├── ROC curves
│   └── summary charts
│
├── train_all_models.py
├── ensemble_bone_density.py
└── README.md



---

## ⚙️ Installation
```bash
pip install torch torchvision scikit-learn matplotlib seaborn pillow numpy
````

---

## ▶️ How to Run

### 🔹 Train Models

```bash
python train_all_models.py
```

### 🔹 Run Ensemble

```bash
python ensemble_bone_density.py
```

---

## 💡 Key Innovation

* Region-based analysis (**Femur vs Tibia**)
* Hybrid approach:

  * Image processing + Deep Learning
* Ensemble learning with AUC weighting
* Clinically meaningful bone-specific predictions

---

## ⚠️ Limitations

* Depends on segmentation quality
* Limited dataset size
* Requires GPU for faster training

---

## 🔮 Future Work

* Real-time web application
* Mobile deployment
* Clinical validation
* Improved segmentation models

 
## ⭐ Conclusion

This project demonstrates a **hybrid ensemble-based system** that improves osteoporosis detection by:

* Separating bone regions
* Using multiple deep learning models
* Combining predictions intelligently

👉 Provides more accurate and clinically useful results



 
