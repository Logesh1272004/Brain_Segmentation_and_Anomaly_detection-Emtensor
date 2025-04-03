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

# Preprocessing

## 1. Preprocessing for Brain Segmentation using 2D UNET

### Dataset Preprocessing:
The preprocessing steps for the **3D MRI scans** focus on extracting 2D slices and normalizing the images for the **2D UNET model**.

### Steps:
1. **Loading Data:**
   - MRI scans in **NIfTI format** (`.nii`) are loaded using the **`nibabel`** library.
2. **Extracting 2D Slices:**
   - The 3D volume is sliced into individual 2D slices along the z-axis (depth).
3. **Data Augmentation:**
   - Random transformations are applied to the data to augment the training set and improve model generalization.
   - Augmentations may include rotation, flipping, and intensity adjustments.
4. **Normalization:**
   - Images are normalized to a range suitable for training, with pixel intensity values typically normalized to [0, 1].
5. **Resizing:**
   - Each 2D slice is resized to a specific size (e.g., **128x128 pixels**) for input into the UNET model.

---

## 2. Preprocessing for Brain Anomaly Detection using EfficientNet-B0

The preprocessing steps for the **MRI images** are designed to prepare the data for efficient training of the **EfficientNet-B0** model.

### Steps:
1. **Loading Data:**
   - MRI images (typically in **JPEG** or **PNG** format) are loaded into memory.
2. **Resizing:**
   - Each image is resized to **224x224 pixels** to match the input size of **EfficientNet-B0**.
3. **Data Augmentation:**
   - Augmentations are applied to the training set to improve the model's ability to generalize:
     - Random horizontal flipping
     - Random rotations
     - Adjusting brightness/contrast
4. **Normalization:**
   - Pixel values are normalized using **ImageNet statistics**:
     - **Mean**: [0.485, 0.456, 0.406]
     - **Std Dev**: [0.229, 0.224, 0.225]
5. **Dataset Split:**
   - The dataset is divided into training and testing sets, where training images are used to train the model and testing images are used for evaluation.

---

## 3. Preprocessing for Brain Segmentation using Gaussian Mixture Model (GMM)

For **brain tissue segmentation** using **Gaussian Mixture Models (GMM)**, the preprocessing steps focus on transforming the 2D MRI images into a suitable format for unsupervised learning.

### Steps:
1. **Loading Data:**
   - 2D MRI images (in **JPEG**, **PNG**, or similar formats) are loaded using the **`skimage`** library.
2. **Grayscale Conversion:**
   - The MRI images are converted to **grayscale** for simplifying segmentation.
3. **Flattening:**
   - The 2D grayscale image is **flattened** into a 1D array, which is required for applying GMM.
4. **Gaussian Mixture Model (GMM):**
   - A **Gaussian Mixture Model** is fitted to the 1D flattened image.
   - GMM is used to classify the image pixels into **three clusters**:
     - **Gray matter**
     - **White matter**
     - **Cerebrospinal Fluid (CSF)**
5. **Reshaping for Visualization:**
   - After the GMM classification, the pixel labels are reshaped back into the original image dimensions for visualization purposes.

---  
# Model Selection

## 1. Model for Brain Segmentation using 2D UNET

### Model: **2D UNET**

For the brain segmentation task, a **2D UNET** architecture is used. UNET is a popular deep learning model architecture for image segmentation, especially in medical imaging tasks. The 2D version of the UNET is applied to **2D slices** extracted from 3D MRI scans.
![U-net_Arc](./Images/u-net-architecture.png)

### Key Features:
- **Encoder-Decoder Architecture**: The model consists of an encoder that captures context and a decoder that enables precise localization.
- **Skip Connections**: Skip connections are used to pass high-resolution features from the encoder to the decoder, preserving spatial information.
- **Loss Function**: **Dice Loss** is used to optimize the segmentation performance. This loss function is specifically designed for imbalanced segmentation tasks like medical imaging, where the target regions (e.g., brain tissues) are relatively small compared to the entire image.

### Training:
- The **UNET** model is trained using the **MONAI** library, which provides ready-to-use implementations for medical image analysis.
- **Optimizer**: Adam optimizer is used to update the model's weights during training.
- **Batch Size**: A batch size of 16 or 32 is typically used depending on available resources.

---

## 2. Model for Brain Anomaly Detection using EfficientNet-B0

### Model: **EfficientNet-B0**

For anomaly detection, the **EfficientNet-B0** model is employed. EfficientNet is a state-of-the-art convolutional neural network (CNN) architecture that is highly efficient and effective for image classification tasks.
![Effi_Arc](./images/output_gmm_segmented_no437.png)

### Key Features:
- **Transfer Learning**: EfficientNet-B0 is used with **pre-trained weights** from the ImageNet dataset, leveraging transfer learning to boost performance.
- **Efficient Architecture**: The EfficientNet architecture scales efficiently and uses a compound scaling method to balance network depth, width, and resolution.
- **Loss Function**: **Cross-Entropy Loss** is used as the loss function to optimize for classification tasks.
- **Activation Function**: **Softmax** is used in the final layer for multi-class classification (e.g., Alzheimer's Disease, Tumor, Cognitively Normal, etc.).

### Training:
- The model is trained using the **Adam optimizer** with a learning rate of **1e-4**.
- A typical batch size of **32** is used, and the model is trained for about **10 epochs**.
- The dataset is resized to **224x224 pixels** to match the input size of EfficientNet-B0.

---

## 3. Model for Brain Segmentation using Gaussian Mixture Model (GMM)

### Model: **Gaussian Mixture Model (GMM)**

For brain tissue segmentation, the **Gaussian Mixture Model (GMM)** is used. This is an unsupervised machine learning model that assumes all data points (in this case, image pixels) are generated from a mixture of several Gaussian distributions.
![Gau_Arc](./images/output_gmm_segmented_no437.png)

### Key Features:
- **Unsupervised Learning**: GMM is an unsupervised model, meaning it does not require labeled data for training. It clusters the data into different categories based on pixel intensity.
- **Clustering**: GMM assumes that the image consists of multiple distinct regions (e.g., gray matter, white matter, and CSF). The model fits a Gaussian distribution to each of these regions and classifies pixels into the corresponding cluster.
- **Number of Clusters**: For brain segmentation, GMM is configured with **3 clusters** corresponding to the three major tissue types in the brain:
  - **Gray Matter**
  - **White Matter**
  - **Cerebrospinal Fluid (CSF)**

### Training:
- The **Gaussian Mixture Model** is fit to the pixel intensity values of the MRI images.
- The model is then used to classify each pixel into one of the three brain tissue types.
- After clustering, the results are reshaped into the original 2D image format for visualization.

---
# Output for Brain Segmentation using 2D UNET

### Output: **Segmented Brain MRI Images**

The output of the **2D UNET** model is a **segmented version** of the input MRI images. The model produces a mask highlighting the segmented regions, such as **gray matter**, **white matter**, and **CSF** (cerebrospinal fluid).


### Explanation:
- The segmented images are saved in **PNG format**, where the regions of interest (gray matter, white matter, CSF) are highlighted in different colors.
- These masks are overlayed on the original MRI slices to visualize the segmentation.
- The **Dice Score** or other performance metrics can be calculated by comparing the predicted masks with ground truth annotations.

### Example Output Visualization:

- **Original MRI Image** vs **Segmented Image**:

  ![Original vs Segmented Example](./images/output_segmented_subject_001.png)

- **Ground Truth** (if available) and **Predicted Segmentation Mask**: These are compared to visualize the model’s accuracy.

  ![Ground Truth vs Predicted Segmentation](./images/output_segmented_subject_002.png)

# Output for Brain Anomaly Detection using EfficientNet-B0

### Output: **Classification Results** (Normal vs Anomalous)

The output of the **EfficientNet-B0** model is a **classification result** for each MRI image in the test set, identifying the condition (e.g., **Alzheimer's Disease**, **Cognitively Normal**, **Tumor**, etc.).

### Example Output Files:
![Confusion Matrix Example](./images/output_confusion_matrix.png)
![Loss/Accuracy Curve Example](./images/output_loss_accuracy_curve.png)

# Output for Brain Segmentation using Gaussian Mixture Model (GMM)

### Output: **Segmented Brain Tissue Types** (Gray Matter, White Matter, CSF)

The output of the **Gaussian Mixture Model (GMM)** is a segmented version of the MRI image where the brain tissues (gray matter, white matter, and CSF) are classified into different regions.

### Example Output Files:

### Explanation:
- The GMM model assigns each pixel in the MRI image to one of the three tissue types: **gray matter**, **white matter**, or **CSF**.
- The output images are saved in **PNG format**, where each pixel is labeled with its respective tissue type (often represented by different colors).
- These segmented outputs are visualized by overlaying the tissue classification on the original MRI slice.

### Example Output Visualizations:

- **Original MRI Image** vs **Segmented Image**: The segmented image shows the regions identified as **gray matter**, **white matter**, and **CSF**.

  ![Original vs Segmented Example](./images/output_gmm_segmented_no437.png)

- **Clustered Image**: The image is divided into clusters, each representing a different tissue type. The final segmentation helps visualize the brain’s tissue structure.

  ![Clustered Image Example](./images/output_gmm_segmented_y232.png)

### Example Directory Structure for Output Files:









