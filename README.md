# Overview

This ASL Detector is a cutting-edge AI-powered application that uses computer vision and deep learning to recognize and classify American Sign Language (ASL) characters in real-time. This application utilizes the device's camera to capture hand landmarks and coordinates, which are then processed by a deep learning model  ( MLP )to identify the corresponding ASL character.

<p align="center">
   <img src="https://github.com/AkramOM606/American-Sign-Language-Detection/assets/162604610/6945d009-8aa7-4bf7-99f8-9743662c5248" width="50%">
</p>

# Usage
By default, when you launch app.py, the inference mode is active. It can also be manually activated in other modes by pressing “n”.

<p align="center">
   <img src="https://github.com/AkramOM606/American-Sign-Language-Detection/assets/162604610/16ed949f-5aa8-4ed4-b49e-a7eb365c8923" width="60%">
</p>

# Table of Contents

1. [Features](#Features)
2. [Requirements](#Requirements)
3. [Installation](#Installation)
4. [Model Training](#Model-Training)


# Features

- **Real-time ASL detection using the device's camera.**
- **Accurate classification of ASL characters using a deep learning model.**
- **Hand landmark tracking for precise gesture recognition.**
- **Support for a wide range of ASL characters and phrases.**
- **High accuracy and robustness in varying lighting conditions.**

 
 # requirements.txt 


Package	Used For
mediapipe	Detecting 21 hand landmarks in real time
opencv-python	Capturing video from webcam and displaying the detection window
tensorflow	Building and training the MLP model to classify ASL signs
numpy	Numerical operations, e.g., feature vector manipulation
pandas	Handling CSV dataset (landmarks and labels)
sklearn	Label encoding, train-test splitting, evaluation
pyttsx3	Text-to-speech for audio output (in GUI app)
tkinter	GUI interface for the ASL detection application
matplotlib	Visualizing model performance or data (in the notebook)


# Installation:

1. Clone the Repository:

```
git clone https://github.com/aman420xd/American-Sign-Language-Detection.git
cd American-Sign-Language-Detection
```

3. Install Dependencies:

```
pip install -r requirements.txt
```

4. Run the Application:

```
python main.py
```
Model Training 
The model is trained in the Jupyter notebook:
📄 keypoint_classification.ipynb

It uses MediaPipe to extract 21 hand landmarks (each having x, y, z coordinates), and then uses a Multi-Layer Perceptron (MLP) model (built with TensorFlow/Keras) to classify the hand pose into ASL characters.

