# Covid-19-Mask-Detection  

## Description  

The **Covid-19 Mask Detection** system is a real-time application designed to monitor and detect individuals wearing face masks in compliance with Covid-19 safety guidelines.  
Using a **ResNet-34** deep convolutional neural network, the model classifies live camera feeds and displays the detection results instantly on the screen.  
This system is built with **PyTorch** for the deep learning backend and **OpenCV** for handling video streams and real-time visualization.

## Table of Contents  

* [Introduction](#introduction)  
* [Idea Behind](#idea-behind)  
* [Prerequisites](#prerequisites)  
* [Installation and Setup](#installation-and-setup)  
* [Results](#Results)  
* [Acknowledgements](#acknowledgements)  

## Introduction  

The **Covid-19 Mask Detection** model uses ResNet-34, a well-known deep learning architecture, to process video streams and classify individuals as either wearing a mask or not.  
This application is especially useful for public spaces like gate entrances, ensuring safety measures are adhered to.  

The real-time system leverages the speed and accuracy of ResNet-34, combined with the power of OpenCV, to create an efficient and effective solution.

## Idea Behind  

1. **Preprocessing the Video Feed:**  
   - Frames from a live camera feed are captured and preprocessed to match the input size expected by ResNet-34.  

2. **Model Inference:**  
   - The ResNet-34 model predicts whether a person in the frame is wearing a mask or not.  

3. **Result Visualization:**  
   - OpenCV overlays bounding boxes and labels ("Mask" or "No Mask") on the live video feed to provide immediate feedback.  

## Prerequisites  

- **Python:** Ensure Python 3.8 or later is installed.  
- **PyTorch:** Required for implementing ResNet-34; install version 1.8.0 or later.  
- **OpenCV:** Required for handling live video feeds.  
- **Camera Feed:** A webcam or laptop camera to provide the input stream. 

## Installation and Setup

1. **Clone the Repository:**

	```sh
	git clone https://github.com/OmarEl-Khashab/Covid-19-Mask-Detection.git
	cd Mask_Detection
	```

2.  **Add your dataset paths :**

	Add your images path in the Training.py:

	```
	data_dir = "/Mask Detection"
	```
3. Start Training your Model by running:

 	```sh
	  python train.py
	```
4. Start Live video inference:

 	```sh
	  python live_video.py
	```
## Results 
Check the Results:

| No Mask Image           | Mask Image            |
|-------------------------|-----------------------|
| ![Image 1](/results%20/nomask.png) | ![Image 2](/results%20/mask.png) |

Project video: https://www.linkedin.com/feed/update/urn:li:activity:6756553053064728576/

## Acknowledgement

This is a big thanks for (https://github.com/AbdallaGomaa) for guidance in my is Machine learning Projects.

Feel free to contribute reach out for the project by opening issues or submitting pull requests. If you have any questions, contact me at omar_khashab11@hotmail.com
