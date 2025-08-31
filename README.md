![Python](https://img.shields.io/badge/python-3.7%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)



🌱 Plant Disease Prediction
  A deep learning project for classifying plant diseases using Convolutional Neural Networks (CNNs).
  This project helps farmers and researchers detect plant diseases from leaf images, using a trained model deployed with Streamlit for easy usage.

Project Structure
  Plant_disease_prediction/
  │── app/                         # Streamlit app files
  │── model_training_notebook/     # Jupyter notebooks for training
  │── test_images/                 # Sample test images
  │── Plant-Model/                 # Trained models & class indices
  │   ├── plant_disease_prediction_model.h5
  │   ├── class_indices.json
  │   ├── model-training.py
  │── README.md

Features
  1. Trains a CNN on the PlantVillage Dataset
  2. Supports multiple plant types and disease categories.
  3. Interactive Streamlit UI for uploading images and predicting diseases.
  4. Provides real-time classification with accuracy and confidence scores.

Installation
 1. Clone the Repository
    git clone https://github.com/shraddhagreddy/Plant_disease_prediction.git
    cd Plant_disease_prediction
 2️. Install Dependencies
    It is recommended to use a virtual environment.
    pip install -r requirements.txt
 3️. Download Dataset
    Download the PlantVillage Dataset and place it inside your working directory.

Model Training
    Run the training script to build and train the CNN: python Plant-Model/model-training.py
    This will:
    Train the CNN on the dataset.
    Save the trained model as plant_disease_prediction_model.h5.
    Generate class_indices.json for mapping predictions.

Running the Streamlit App
    Once the model is trained, run the web app with: streamlit run app/app.py
    Upload a leaf image (.jpg, .jpeg, .png)
    Get instant disease predictions

Example Results
    Model Accuracy & Loss
    The CNN achieves ~92% validation accuracy (update after training).

Sample Prediction
    Upload a leaf image and get prediction:
    Prediction: Apple Scab 

Requirements
    Python 3.7+
    TensorFlow / Keras
    NumPy, Pandas
    Matplotlib, Pillow
    Streamlit

Future Improvements
    Add Grad-CAM visualizations for explainable AI.
    Deploy model on cloud (AWS/GCP/Heroku).
    Improve accuracy with transfer learning (ResNet, EfficientNet).

Contributing
    Contributions are welcome!
    Fork the repo
    Create a new branch
    Submit a Pull Request

License
    This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
