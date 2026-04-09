🎭 DeepX – Deepfake Video Detection System

DeepX is a deep learning–based **Deepfake Video Detection project** that detects whether a given video is **REAL or FAKE** and provides **visual explainability** using CNN-based attention (Grad-CAM–style heatmaps).

This project focuses on **both accuracy and explainability**, making the model’s decisions interpretable for users.


## Project Overview

Deepfake videos are artificially generated or manipulated videos that can spread misinformation and cause serious harm.  
DeepX aims to detect such videos by:

- Extracting face frames from videos
- Learning spatial features using a pre-trained CNN
- Learning temporal patterns across frames using LSTM
- Providing explainability heatmaps to highlight suspicious regions


## Model Architecture for video
Video → Face Frames → CNN (Xception) → Feature Vectors
→ LSTM → Binary Classification (Real / Fake)



### Key Components:
- **Face Detection:** MTCNN + Haar Cascade (fallback)
- **Feature Extraction:** Pre-trained Xception (ImageNet)
- **Temporal Modeling:** LSTM
- **Classification:** Sigmoid-based binary output
- **Explainability:** CNN activation-based Grad-CAM heatmaps


## Technologies Used

- **Python**
- **TensorFlow & Keras**
- **OpenCV**
- **MTCNN**
- **NumPy**
- **Matplotlib**
- **Google Colab**
- **GitHub**

---

## Dataset

- Trained on **real and fake videos** (balanced dataset)
- Each video is processed into **30 face frames**
- Faces are resized to **299 × 299** pixels
- Dataset can be replaced with:
  - FaceForensics++
  - DFDC
  - Custom datasets


## Explainability (Grad-CAM)

The project provides **visual explanations** for predictions:

- Highlights **suspicious facial regions**
- Helps understand *why* a frame is classified as fake
- Displays explainability for **selected frames only** (user-friendly)

> Note: Since the final decision is made by an LSTM, explainability is generated using **CNN activation-based attention maps**.


## Output Example

- **Prediction:** REAL / DEEPFAKE  
- **Confidence Score:** Percentage  
- **Explainability:** Heatmaps over facial regions  


## How to Run

1. Clone the repository:
   git clone https://github.com/your-username/DeepX.git


📜 License
This project is for educational and research purposes only.


