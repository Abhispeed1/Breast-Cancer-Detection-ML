# 🎀 Breast Cancer Detection using Machine Learning Classifiers

## 🎯 Project Overview

This repository hosts a machine learning project focused on **predicting breast cancer malignancy** based on features derived from digitized images of fine needle aspirate (FNA) of breast masses. The goal is to develop and evaluate various classification models to achieve high accuracy in distinguishing between **Malignant** (cancerous) and **Benign** (non-cancerous) samples.

---

## 📊 Dataset

The project utilizes the **Wisconsin Breast Cancer (Diagnostic) Dataset**, which contains 569 instances with 30 numeric features computed from the images, such as radius, texture, perimeter, and area of the cell nuclei.

### Key Features Used:

* **Mean:** Mean of the features.
* **Standard Error (SE):** Standard error of the features.
* **Worst:** "Worst" or largest (mean of the three largest values) of the features.

---

## 🛠️ Technology and Methodology

This project is implemented using Python and relies heavily on the **Scikit-learn** library for model building and evaluation.

### Key Classifiers Evaluated:

The following popular machine learning algorithms were implemented and compared:

1.  **K-Nearest Neighbors (KNN)**
2.  **Support Vector Machine (SVM)**
3.  **Logistic Regression**
4.  **Decision Tree**
5.  **Random Forest**

### Methodology Steps:

1.  **Data Loading and Exploration:** Initial review and understanding of the dataset structure.
2.  **Preprocessing:** Handling missing values (if any) and **Feature Scaling** to normalize input data.
3.  **Model Training:** Training each classifier on the training set.
4.  **Model Evaluation:** Assessing performance using metrics such as **Accuracy**, **Precision**, **Recall**, and the **F1-Score**.

---

## 🏃 Getting Started

To run this project locally, you will need to have Python and the necessary libraries installed.

### Prerequisites

* Python 3.x
* Jupyter Notebook (or JupyterLab)

### Installation

1.  Clone this repository:
    ```bash
    git clone [https://github.com/Abhispeed1/Breast-Cancer-Detection-ML.git](https://github.com/Abhispeed1/Breast-Cancer-Detection-ML.git)
    cd Breast-Cancer-Detection-ML
    ```
2.  Install the required Python packages (create a `requirements.txt` if necessary, or install individually):
    ```bash
    pip install numpy pandas scikit-learn matplotlib seaborn jupyter
    ```
3.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook Breast_Cancer_Detection_Using_Machine_Learning_Classifier-checkpoint.ipynb
    ```

Run the cells in the notebook sequentially to see the data processing, model training, and performance results.

---

## 🤝 Contact

If you have any questions or suggestions, feel free to reach out!

* **GitHub:** [@Abhispeed1](https://github.com/Abhispeed1)
