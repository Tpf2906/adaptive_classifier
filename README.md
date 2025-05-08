# User preference based adaptive classifier

This project focuses on training classifiers on image features extracted using the CLIP model, applying PCA for dimensionality reduction, and using an ensemble of different classifiers for image and text classification. The entire pipeline involves several steps:

1. **Dataset Preparation**: Prepare a dataset with image captions and corresponding class labels.
2. **Feature Extraction**: Use the CLIP model to extract features from images and captions.
3. **PCA & Standardization**: Standardize the extracted features and apply PCA for dimensionality reduction.
4. **Classification**: Train multiple classifiers on the features and combine them using ensemble methods.

## Project Structure

### 1. `get_dataset_with_classes.ipynb`
This notebook processes the dataset and generates a JSON file that contains image captions and corresponding class labels. The JSON file is used as input for the feature extraction step.

- **Input**: A dataset of images with captions and class labels.
- **Output**: A JSON file with image filenames, captions, and class IDs.

### 2. `extract_clip_features.ipynb`
This notebook utilizes the CLIP (Contrastive Language-Image Pretraining) model from OpenAI to extract image and text features. These features are stored in the `clip_features` folder.

- **Input**: The JSON file generated in `get_dataset_with_classes.ipynb`.
- **Output**: Extracted image and text features are saved in the `clip_features` folder.

### 3. `pca_variance.ipynb`
This notebook performs feature standardization, applies PCA (Principal Component Analysis) on the extracted features, and saves the standardized features, the PCA model, and the scaler.

- **Input**: Extracted features from the `clip_features` folder.
- **Output**: PCA-transformed features and saved models (`scaler` and `pca`).

### 4. `classifiers/`
This directory contains various classifiers that can be trained on the processed features from the previous steps. The classifiers include:

- **SVM** (Support Vector Machine)
- **Random Forest**
- **Logistic Regression**
- **Decision Trees**
- **K-Nearest Neighbors (KNN)**
- **Naive Bayes**
- **Gradient Boosting Machine (GBM)**
- **AdaBoost**
- **XGBoost**
- **LDA (Linear Discriminant Analysis)**

These classifiers are designed to be trained using the features extracted by CLIP and reduced in dimensionality by PCA. The classifiers can be used individually or combined using an ensemble method.

### 5. `requirements.txt`
This file lists the necessary Python packages and dependencies needed to run the project. You can install the required packages using:

```bash
pip install -r requirements.txt
