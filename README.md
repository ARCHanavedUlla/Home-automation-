# Home-automation-
This project implements a smart home automation system that uses machine learning to predict and automate home devices' actions based on sensor data such as temperature, humidity, light intensity, and motion detection. The system learns user preferences and environmental conditions to optimize energy usage and enhance comfort.

Workflow: 

<img width="636" height="392" alt="the" src="https://github.com/user-attachments/assets/6971c3a0-28f2-4fb8-aa4e-7e16318b0d40" />

<img width="862" height="507" alt="Picture1" src="https://github.com/user-attachments/assets/1a3df8af-65e3-4889-80fd-daf67ef08c82" />




Key Features

Collect and preprocess sensor data from smart home devices.
Train multiple machine learning models (KNN, SVM, Random Forest, MLP, CNN) for predicting device control actions.
Real-time device control and automation based on model predictions.
User-friendly web interface for monitoring and manual override.
Visualization of sensor data and automation status.
Installation:

Clone the repository:

1.git clone https://github.com/ARCHanavedUlla/home-automation-ml.git

Install required dependencies:

2.pip install -r requirements.txt

Project Structure



home-automation-ml/
│
├── data/                   # Sensor datasets (raw and processed)
├── models/
│   ├── train_models.py     # Train machine learning models (KNN, SVM, RF, MLP, CNN)
│   └── evaluate.py         # Evaluation metrics and feature importance plots
├── scripts/
│   ├── preprocess.py       # Data cleaning and preprocessing
│   ├── automate.py         # Real-time automation logic using trained models
├── templates/
│   ├── index.html          # Dark-themed UI for sensor input and control panel
│   └── result.html         # Displays automation recommendations/status
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
