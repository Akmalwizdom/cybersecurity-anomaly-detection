
# Cybersecurity Anomaly Detection

## Project Overview
This project implements a Machine Learning-based system for detecting anomalies in network traffic, specifically designed for cybersecurity applications. It utilizes the K-Means Clustering algorithm to categorize network patterns into "Normal" and "High Risk" segments based on anomaly scores.

The system includes a complete pipeline from data processing and model training to a web-based threat intelligence dashboard for real-time analysis.

## Key Features
- **Unsupervised Learning**: Utilizes K-Means clustering (k=2) to automatically identify potential threats without labeled data.
- **Web Dashboard**: Interactive Flask-based interface for monitoring and analyzing traffic.
- **REST API**: JSON-based endpoint for integrating the detection model with other security tools.
- **Visualization Suite**: Automated generation of high-quality charts including Scatter Plots, PCA Projections, and Elbow Analysis.
- **Scalable Architecture**: Modular design separating model training, visualization, and deployment.

## Technology Stack
- **Language**: Python 3.x
- **Web Framework**: Flask
- **Machine Learning**: Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Model Persistence**: Joblib

## Installation

### Prerequisites
Ensure you have Python 3.8 or higher installed on your system.

### Steps
1. Clone the repository to your local machine.
2. Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage Guide

### 1. Model Training
Initialize and train the K-Means clustering model using the provided dataset. This step performs feature selection, scaling, and model serialization.

```bash
python train_model.py
```

**Output**: Trained models will be saved in the `models/` directory.

### 2. Generate Visualizations
Create a suite of analytical charts to evaluate the model's performance and cluster distribution.

```bash
python create_visualizations.py
```

**Output**: Visualizations (PNG files) will be generated in the `visualizations/` directory.

### 3. Launch Dashboard
Start the web application to interact with the model.

```bash
python app.py
```

Access the dashboard in your browser at: `http://localhost:5000`

## Project Structure

```text
cybersecurity-anomaly-detection/
├── app.py                      # Flask application entry point
├── train_model.py              # ML pipeline for training and saving models
├── create_visualizations.py    # Script to generate analytical charts
├── requirements.txt            # Python dependencies
├── dataset/
│   └── cybersecurity_attacks.csv
├── models/                     # Directory for saved models and scaler
├── static/                     # CSS and static assets
├── templates/                  # HTML templates
└── visualizations/             # Output directory for generated charts
```

## Model Details
The system employs **K-Means Clustering** with an optimal cluster count of **k=2**, determined through Elbow Method and Silhouette Analysis.

- **Cluster 0**: Represents normal traffic patterns with low anomaly scores.
- **Cluster 1**: Represents high-risk traffic indicative of potential cyber attacks.

The model achieves a Silhouette Score of approximately **0.62**, indicating a strong separation between normal and anomalous traffic clusters.
