![Python](https://img.shields.io/badge/python-3.7%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)



🌱 Plant Disease Prediction

Plant Disease Prediction is a deep learning project designed to classify plant diseases from leaf images using Convolutional Neural Networks (CNNs).
It provides a simple Streamlit web app where users can upload an image of a leaf and instantly get a prediction about the plant’s health.

This project aims to assist farmers, researchers, and agricultural experts in early disease detection to improve crop yield and reduce losses.

✨ Features

🧠 CNN-based model trained on the PlantVillage Dataset

📂 Supports multiple plant types and disease categories

📸 Image upload & classification via Streamlit

⚡ Real-time prediction with accuracy & confidence scores

📊 Training history visualization (accuracy & loss plots)

💾 Model + class mapping saved for easy deployment

📂 Project Structure
Plant_disease_prediction/
│── app/                         # Streamlit app files
│── model_training_notebook/     # Jupyter notebooks for training
│── test_images/                 # Sample test images
│── Plant-Model/                 # Trained models & class indices
│   ├── plant_disease_prediction_model.h5
│   ├── class_indices.json
│   ├── model-training.py
│── README.md
│── LICENSE

🚀 Getting Started
✅ Prerequisites

Python 3.7+

TensorFlow / Keras

NumPy, Pandas

Matplotlib, Pillow

Streamlit

⚙️ Setup Instructions

Clone the repository:

git clone https://github.com/shraddhagreddy/Plant_disease_prediction.git
cd Plant_disease_prediction


Install dependencies:

pip install -r requirements.txt


Download the PlantVillage Dataset
 and place it inside your working directory.

🏋️ Model Training

Run the training script:

python Plant-Model/model-training.py


This will:

Train the CNN on the dataset

Save the trained model as plant_disease_prediction_model.h5

Generate class_indices.json for mapping predictions

🌐 Running the Streamlit App

Run the web app with:

streamlit run app/app.py


Upload a leaf image (.jpg, .jpeg, .png)

Get instant disease predictions 🎉

📊 Example Results
Training Performance

Model achieves ~92% validation accuracy (update after training).




Sample Prediction

Upload a leaf image →
Prediction: Apple Scab 🍏🍂

🔮 Future Improvements

Add Grad-CAM visualizations for explainable AI

Deploy model to cloud (AWS/GCP/Heroku)

Improve accuracy with transfer learning (ResNet, EfficientNet)

Add mobile app integration for farmers

🛠 Tech Stack

Language: Python

Frameworks: TensorFlow, Keras, Streamlit

Data: PlantVillage Dataset

Libraries: NumPy, Pandas, Matplotlib, Pillow

👩‍💻 Author

Shraddha Reddy
📧 shraddhagreddy@gmail.com

💡 Always open to collaboration & feedback!

📜 License

This project is licensed under the MIT License © 2025 Shraddha Reddy.
See the LICENSE
 file for details.
