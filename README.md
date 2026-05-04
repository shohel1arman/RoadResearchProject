# Deep Learning for Road Surface Damage Classification, Detection, and Reporting Using YOLOv12 and LLaMA

# 🛣️ AI-Powered Road Damage Detection & Assessment System

A comprehensive solution for automated road inspection. This project combines **Deep Learning Classification**, **Real-time Object Detection (YOLO)**, and **Generative AI (LLMs)** to detect, analyze, and generate professional reports on road defects like potholes, cracks, and surface erosion.

## 🚀 Key Features

* **Hybrid AI Analysis**:
    * **Classification**: Classifies images into *Crack*, *Pothole*, or *Surface Erosion* using custom-trained CNNs and Transfer Learning models.
    * **Object Detection**: Localizes and counts multiple defects per image using **YOLOv8** and **YOLOv12** (see `models/YOLOv8.ipynb` and `models/YOLOv12.ipynb`).
* **Smart Quality Control**: Automatically detects lighting conditions (e.g., "Very Low Light", "Bright Light") to ensure image suitability before processing.
* **Generative AI Reporting**: Integrates **Llama 3.2** (via Ollama) to generate human-readable, professional municipal reports based on damage severity scores.
* **Live Video Streaming**: Supports real-time inference on video streams (e.g., IP Webcam) with MJPEG streaming of annotated frames.
* **REST API**: A fully functional Django Backend to serve mobile or web frontends.

## 🏗️ System Architecture

The backend is built with **Django & Django REST Framework**. It orchestrates a pipeline of three distinct AI components:

1.  **Preprocessor**: Checks image brightness/quality.
2.  **Classifier**: Runs a TensorFlow model (MobileNet/CNN) to determine the primary defect type.
3.  **Detector & Reporter**: Runs YOLO to count defects, calculates a severity score, and prompts a local LLM to write a summary.

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

# 🛠️ Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Frankenstein452/RoadResearchProject
```
## 2. Go to the Agent Directory

```bash
cd RoadResearchProject
```
## 3. Set Up Environment

```bash
python -m venv venv

# windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Upgrade PIP version

```bash
python.exe -m pip install --upgrade pip
```

## 5. Run ollama

```bash
# open cmd terminal & activate ollama
ollama run llama3.2
```

## 6. Environment Configuration
Add paths to your trained models in .env file:

```bash
echo "MODEL_PATH=path/to/your/mobilenet_model.h5" >> .env
echo "MODEL_PATH2=path/to/your/yolo_best.pt" >> .env
echo "SECRET_KEY=your_django_secret_key" >> .env
echo "DEBUG=True" >> .env
```

## 7. Database Setup

```bash
python manage.py makemigrations
python manage.py migrate
```

## 8. Run the Server
Ensure your Ollama instance is running (Step 5), then start the Django server:

```bash
python manage.py runserver
```