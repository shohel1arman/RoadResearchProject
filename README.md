# Deep Learning for Road Surface Damage Classification, Detection, and Reporting Using YOLOv12 and LLaMA

# 🛣️ AI-Powered Road Damage Detection & Assessment System

A comprehensive solution for automated road inspection. This project combines **Deep Learning Classification**, **Real-time Object Detection (YOLO)**, and **Generative AI (LLMs)** to detect, analyze, and generate professional reports on road defects like potholes, cracks, and surface erosion.

## 🚀 Key Features

* **Hybrid AI Analysis**:
    * **Classification**: Classifies images into *Crack*, *Pothole*, or *Surface Erosion* using custom-trained CNNs and Transfer Learning models.
    * **Object Detection**: Localizes and counts multiple defects per image using **YOLOv8**.
* **Smart Quality Control**: Automatically detects lighting conditions (e.g., "Very Low Light", "Bright Light") to ensure image suitability before processing.
* **Generative AI Reporting**: Integrates **Llama 3.2** (via Ollama) to generate human-readable, professional municipal reports based on damage severity scores.
* **Live Video Streaming**: Supports real-time inference on video streams (e.g., IP Webcam) with MJPEG streaming of annotated frames.
* **REST API**: A fully functional Django Backend to serve mobile or web frontends.

## 🏗️ System Architecture

The backend is built with **Django & Django REST Framework**. It orchestrates a pipeline of three distinct AI components:

1.  **Preprocessor**: Checks image brightness/quality.
2.  **Classifier**: Runs a TensorFlow model (MobileNet/CNN) to determine the primary defect type.
3.  **Detector & Reporter**: Runs YOLO to count defects, calculates a severity score, and prompts a local LLM to write a summary.

## 📊 Model Performance

Various architectures were trained and benchmarked for the classification task. **MobileNet** was selected as the production model due to its superior accuracy and efficiency.

| Model Architecture | Test Accuracy | F1 Score (Macro) | Status |
| :--- | :--- | :--- | :--- |
| **MobileNet** | **95.57%** | **0.95** | 🚀 **Deployed** |
| Custom CNN | 93.29% | 0.91 | Excellent Backup |
| VGG19 | 84.82% | - | Overfitting Observed |
| ResNet | 55.74% | 0.43 | Underperforming |
| EfficientNet | 53.75% | - | Underperforming |

*Note: Object Detection is handled by a custom-trained YOLOv8 model achieving ~0.60 mAP50.*

## 🛠️ Tech Stack

* **Backend Framework**: Django 5.2, Django REST Framework
* **Machine Learning**: TensorFlow/Keras, PyTorch, Ultralytics (YOLO)
* **Generative AI**: Ollama (Llama 3.2)
* **Computer Vision**: OpenCV, Pillow
* **Database**: SQLite (Dev), configurable for PostgreSQL
* **Tools**: NumPy, Pandas, Matplotlib

## ⚙️ Installation & Setup

### Prerequisites
* Python 3.10+
* [Ollama](https://ollama.com/) installed and running (for GenAI features).
* Git

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/road-damage-detection.git](https://github.com/yourusername/road-damage-detection.git)
cd road-damage-detection

pip install -r requirements.txt