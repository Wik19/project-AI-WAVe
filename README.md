# Project AI-WAVe 🚁🔊

**Advanced UAV Fault Detection via Audio Analysis**

Project AI-WAVe uses deep learning to diagnose and detect faults in Unmanned Aerial Vehicles (UAVs) by analyzing their acoustic signatures. By leveraging audio data—captured from single or dual-microphone setups—this system can identify mechanical anomalies before they lead to failure.

## 📄 Documentation
For a deep dive into the methodology, architecture, and research behind this project, please refer to our detailed documentation:

👉 **[Read the Full Project Documentation](https://docs.google.com/document/d/1HvZyDjNCPwkZXRRzvsuJbfO-qw1WxypRa7KnO1tnTx4/edit?usp=sharing)**

---

## 🌟 Key Features
* **Acoustic Fault Diagnosis:** Non-invasive health monitoring using sound.
* **Multi-Channel Support:** Includes models optimized for **2-microphone** input arrays for better spatial noise reduction.
* **Deep Learning Core:** Powered by custom Keras models trained on high-resolution WAV datasets.

## 📂 Models
This repository utilizes **Git LFS** to store high-fidelity trained models.
* `uav_fault_model.keras`: Baseline fault detection model.
* `uav_fault_model_02.keras`: Advanced model with 2-mic support (Large File).

*Note: Ensure you have Git LFS installed (`git lfs install`) to pull the full model files.*

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* TensorFlow / Keras
* Git LFS

### Installation
```bash
git clone [https://github.com/Wik19/project-AI-WAVe.git](https://github.com/Wik19/project-AI-WAVe.git)
cd project-AI-WAVe
pip install -r requirements.txt