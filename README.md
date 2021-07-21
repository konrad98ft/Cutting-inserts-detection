# Cutting Inserts Inspection
> Machine vision algorithm utilized for damage inspection of the cutting inserts used in lathe machine. Project utilizes hybrid approach to the image classification by using "clasical" and deep learning algorithms. Program is prepared for smart camera ADLINK NEON 2000 [ more_info ](https://www.adlinktech.com/Products/Deep_Learning_Accelerator_Platform_and_Server/AI_Machine_Vision_Devices/NEON-2000-JT2_Series).

## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Features](#features)
* [Project Development](#project-development)
* [Setup](#setup)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)
<!-- * [License](#license) -->


## General Information
- Project enables vision controll of the cutting inserts usage
- Main pourpose is to provide autonomy of the controlling process
- Hybrid approach - "classical" and deep learning methods
<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


## Technologies Used
- Python 3.6.9
- TensorFlow 2.3.1
- OpenCv 4.3.0
- NumPy 1.18.5
- SciPy 1.6.2
- JetPack 4.4 with following software enviornment [ more info here ](https://www.adlinktech.com/Products/Deep_Learning_Accelerator_Platform_and_Server/AI_Machine_Vision_Devices/NEON-2000-JT2_Series).



## Features
List the ready features here:
- Autonomously finding cutting edge
- Detecting breaches of the cutting edge
- Distracting surface contaminations from real damages


## Project Development
![s1](https://user-images.githubusercontent.com/62110076/118273840-ba78ca00-b4c4-11eb-9893-0470e6860e2d.png)

![s2](https://user-images.githubusercontent.com/62110076/118274719-c3b66680-b4c5-11eb-8d37-a3583593b515.png)

![s3](https://user-images.githubusercontent.com/62110076/118274042-f318a380-b4c4-11eb-867d-db85aee0a6ed.png)

![s4](https://user-images.githubusercontent.com/62110076/118274106-0592dd00-b4c5-11eb-9b9d-06a185ca9fb8.png)

![s5](https://user-images.githubusercontent.com/62110076/118274197-1e02f780-b4c5-11eb-97ca-d411389e59c2.png)

## Setup
Project utilizes cutting inserts images collected by ADLINK NEON 2000 camera. Stored:
https://drive.google.com/drive/folders/1lf42JO9PQdO09VsaYtjwALDH3pwM55xS


## Project Status
Project is: _in progress_ 


## Room for Improvement
* Algorithms optimazation. 
* Providing greater independence from lighting.
* Examining other deep learning pre-trained networks
* Feeding deep learning algorithm with syntetic data generated in Blender





 


## Google Colab project for training pourposes
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1q0ud51vhYlfaU56DseQxJFiGECGlMdUl#scrollTo=jFm_l3ABcbT3)

