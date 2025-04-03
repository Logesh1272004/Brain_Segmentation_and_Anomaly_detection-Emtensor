# Datasets

## 1. Dataset for Brain Segmentation using 2D UNET

The dataset for this project consists of **3D brain MRI scans** in the **NIfTI format** (`.nii`), which is commonly used for storing volumetric (3D) medical image data.

### Dataset Structure:
- The dataset contains **3D MRI volumes** for different subjects.
- Each subject has a corresponding `.nii` file representing their MRI scan.




### Explanation:
- The 3D MRI scans are processed by **extracting 2D slices** and passing them through a **2D UNET model** for segmentation.
- The goal is to segment different brain regions such as **gray matter**, **white matter**, and **CSF** (cerebrospinal fluid).

---

## 2. Dataset for Brain Anomaly Detection using EfficientNet-B0

The anomaly detection dataset consists of **brain MRI images** categorized into different **disease classes**. The images are used for classifying the condition of the brain (normal or anomalous) using **EfficientNet-B0**.

### Dataset Structure:
- The dataset is divided into **training** and **testing** sets.
- The training data is organized into subdirectories for each class, and the test set is used for model evaluation.

### Explanation:
- The dataset contains MRI images classified into multiple disease categories, such as:
  - **AD**: Alzheimer's Disease
  - **CN**: Cognitively Normal
  - **EMCI**: Early Mild Cognitive Impairment
  - **MCI**: Mild Cognitive Impairment
  - **Tumor**: Tumor images
- These images are resized to **224x224 pixels** and used for training a **deep learning model** (**EfficientNet-B0**) for classification.

---

## 3. Dataset for Brain Segmentation using Gaussian Mixture Model (GMM)

This dataset consists of **single 2D brain MRI images** used for **tissue segmentation** using an unsupervised **Gaussian Mixture Model (GMM)**.

### Dataset Structure:
- The dataset includes **individual 2D MRI slices** that are loaded for segmentation.

### Explanation:
- The MRI images are first **converted to grayscale** and then **flattened into a 1D vector**.
- **Gaussian Mixture Models (GMM)** are applied to the images to classify pixels into three clusters:
  - **Gray matter**
  - **White matter**
  - **CSF** (Cerebrospinal Fluid)
- The goal is to identify and segment different tissue types in the brain using this **unsupervised method**.


