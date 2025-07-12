
# 🛡️ Credit Card Fraud Detection

A Machine Learning project focused on identifying fraudulent credit card transactions from a highly imbalanced dataset using advanced data preprocessing techniques, oversampling methods, and robust classification models.

---

## 🧠 Problem Statement

Credit card fraud is a significant issue in the financial industry, costing companies and customers billions each year. The key challenge lies in the **highly imbalanced** nature of the data — where fraudulent transactions make up only a tiny fraction of the total data — making standard machine learning approaches ineffective.

This project aims to:
- Build an accurate and interpretable model to identify fraud.
- Mitigate the effect of class imbalance.
- Achieve high recall and precision for the minority (fraud) class.
- Provide business insights for fraud prevention systems.

---

## 📁 Dataset Overview

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Fraudulent Transactions**: 492 (0.172%)
- **Legitimate Transactions**: 284,315 (99.828%)
- **Features**: 30 anonymized features (V1-V28), `Time`, `Amount`, and `Class`

---

## 🧰 Tools & Technologies Used

### 🐍 Python Libraries
| Library | Purpose |
|--------|---------|
| `pandas`, `numpy` | Data loading and manipulation |
| `matplotlib`, `seaborn` | Data visualization |
| `scikit-learn` | Model building, evaluation, preprocessing |
| `imblearn` | SMOTE oversampling |
| `warnings`, `time` | Miscellaneous utilities for clean execution |

### 💻 ML Algorithms Implemented
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors (KNN)

---

## ⚙️ Project Workflow

### 1. 🔍 Exploratory Data Analysis (EDA)
- Inspected class distribution to confirm imbalance.
- Plotted correlation heatmaps to find relationships between anonymized features.
- Visualized distributions of transaction amounts and times across both classes.

### 2. 🧼 Data Preprocessing
- **Feature Scaling**: `StandardScaler` applied to `Amount` and `Time`.
- **Train-Test Split**: Performed an 80-20 split with `stratify=y` to maintain class balance.
- **Oversampling**: Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset.

### 3. 🤖 Model Training & Evaluation
Trained multiple models and evaluated using:
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)
- ROC AUC Score
- Precision-Recall Curve

### 4. 📊 Performance Comparison
| Model | Accuracy | Precision | Recall | F1-score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~94.00% | 0.88 | 0.91 | 0.89 | 0.97 |
| Decision Tree | ~99.90% | 0.96 | 0.90 | 0.93 | 0.97 |
| K-Nearest Neighbors | ~99.87% | 0.93 | 0.87 | 0.90 | 0.95 |
| **Random Forest** | **~99.98%** | **0.99** | **0.96** | **0.98** | **0.999** |

✅ **Random Forest Classifier** significantly outperformed others in terms of both precision and recall, making it the most reliable model for detecting fraudulent transactions.

---

## 📈 Visualizations Included

- Class distribution before and after SMOTE.
- Heatmap showing feature correlations.
- ROC Curves and Precision-Recall Curves for all models.
- Confusion matrices to visualize prediction outcomes.

---

## 💡 Key Insights

- **Data imbalance** poses a major challenge; without handling it, models tend to predict only the majority class.
- **SMOTE** effectively helped balance the classes by generating synthetic samples for the minority class.
- **Random Forest** provides robust and reliable results, balancing recall (for catching fraud) and precision (to reduce false alerts).

---

## 🚀 How to Run This Project

1. **Clone this repository**:
   ```bash
   git clone https://github.com/your-username/Credit-Card-Fraud-Detection.git
   cd Credit-Card-Fraud-Detection
   ```

2. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter notebook**:
   ```bash
   jupyter notebook Credit_card_fraud_detection.ipynb
   ```

---

## 📄 requirements.txt

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
```

---

## 📌 Use Cases

- Real-time fraud detection in payment gateways and banking systems.
- Backend fraud analytics dashboard for financial institutions.
- Alert systems for abnormal financial behavior.

---

## 👨‍💻 Author

**Venkat Peddireddy**  
GitHub: [@Peddireddy2002](https://github.com/Peddireddy2002)  
LinkedIn: *[Add your LinkedIn profile]*

---

## 📃 License

This project is licensed under the [MIT License](LICENSE).
