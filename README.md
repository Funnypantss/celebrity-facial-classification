# Celebrity Image Classifier using ML

## 📌 Project Overview
This project is a machine learning pipeline designed to accurately identify and classify celebrity faces. [cite_start]It processes raw images by isolating critical facial features and converting them to grayscale to standardize inputs and reduce computational noise[cite: 28]. [cite_start]The processed images are then compared against an existing dataset to classify faces based on similarity[cite: 29].

## 🚀 Key Features
* [cite_start]**Targeted Feature Extraction:** Isolates specific facial landmarks, such as the eyes and nose, to improve model focus and accuracy[cite: 28].
* [cite_start]**Data Standardization:** Implements grayscale conversion for all input images to streamline processing[cite: 28].
* [cite_start]**Robust Data Pipeline:** Handles extensive data cleaning and manipulation using Pandas and NumPy to prepare inputs effectively[cite: 27].
* [cite_start]**Optimized Training:** Utilizes an 80-20 split for training and testing data to ensure rigorous model validation.

## 🛠️ Tech Stack
* [cite_start]**Language:** Python [cite: 27]
* [cite_start]**Data Manipulation:** NumPy, Pandas [cite: 27]
* **Machine Learning:** Scikit-learn (for similarity comparison and classification)
* **Image Processing:** OpenCV / PIL (Python Imaging Library)

## ⚙️ Installation & Setup
To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/celebrity-facial-classification-ml.git](https://github.com/yourusername/celebrity-facial-classification-ml.git)
   cd celebrity-facial-classification-ml
