# 🌍 AI-Driven Air Quality Prediction using Hybrid Digital Twin## 📌 OverviewThis project presents a **Hybrid Digital Twin (HDT) framework** for predicting Air Quality Index (AQI) in smart city environments. It combines **environmental simulations** with **deep learning models** to improve prediction accuracy and real-world adaptability.---## 🚀 Key Features* 🔬 Hybrid Digital Twin architecture (Simulation + AI)* 🧠 Deep Learning Models:  * CNN (Spatial features)  * LSTM & GRU (Temporal patterns)  * GNN (Graph-based spatial relationships)* ⚡ Cross-Modal Attention for model fusion* 🔄 Bayesian Updating for continuous learning* 🔍 Explainable AI using SHAP & LIME* 🌆 Smart city AQI forecasting & hotspot detection---## 🏗️ System ArchitectureThe system consists of:1. Data Collection & Preprocessing2. Environmental Simulation   * SimPy (Industrial emissions)   * SUMO (Traffic emissions)3. Multi-Model Learning (CNN, LSTM, GRU, GNN)4. Feature Fusion with Attention5. Bayesian Adaptive Learning6. Explainability Layer---## 📊 Dataset* Source: Central Pollution Control Board (CPCB), Government of India – AQI dataset collected via API* Cities: 26 Indian cities* Time Period: 2019 – 2023* Features:  * PM2.5, PM10  * NO₂, CO, O₃, SO₂---## 📈 Results| Model                 | RMSE      | MAE      | R²        || --------------------- | --------- | -------- | --------- || CNN                   | 22.14     | 15.72    | 0.847     || LSTM                  | 17.83     | 12.11    | 0.891     || GRU                   | 16.92     | 11.34    | 0.902     || GNN                   | 15.88     | 10.40    | 0.913     || **Hybrid (Proposed)** | **13.47** | **9.12** | **0.941** |---## 🛠️ Tech Stack* Python 3.11* PyTorch & PyTorch Geometric* SimPy (Simulation)* SUMO (Traffic modeling)* Scikit-learn* SHAP & LIME---## 📂 Project Structure```aqi-digital-twin-prediction/├── data/│   ├── raw/                  # Original datasets (CPCB, etc.)│   ├── processed/            # Cleaned & preprocessed data├── notebooks/│   ├── data_preprocessing.ipynb│   ├── eda_analysis.ipynb│   ├── model_training.ipynb├── src/│   ├── preprocessing.py│   ├── simulation.py         # SimPy + SUMO logic│   ├── models/│   │   ├── cnn.py│   │   ├── lstm.py│   │   ├── gru.py│   │   ├── gnn.py│   ├── fusion.py            # cross-modal attention│   ├── bayesian_update.py│   ├── explainability.py    # SHAP, LIME├── results/│   ├── graphs/│   ├── metrics.txt│             ├── requirements.txt└── README.md```---## ▶️ How to Run```bashgit clone https://github.com/your-username/aqi-digital-twin-prediction.gitcd aqi-digital-twin-predictionpip install -r requirements.txtpython src/run_all.py```---## 🎯 Applications* Smart city pollution monitoring* AQI forecasting & alerts* Policy simulation (traffic, industrial control)* Environmental decision support---## 🔮 Future Work* IoT integration for real-time data* Transformer-based models* Federated learning across cities* Satellite data integration---## 👩‍💻 Authors* N LAHARI* AMRUTHA VARSHINI P* PARHAVI G.V* VEEKSHITHA P# 🌍 AI-Driven Air Quality Prediction using Hybrid Digital Twin

## 📌 Overview

This project presents a **Hybrid Digital Twin (HDT) framework** for predicting Air Quality Index (AQI) in smart city environments. It combines **environmental simulations** with **deep learning models** to improve prediction accuracy and real-world adaptability.

---

## 🚀 Key Features

* 🔬 Hybrid Digital Twin architecture (Simulation + AI)
* 🧠 Deep Learning Models:

  * CNN (Spatial features)
  * LSTM & GRU (Temporal patterns)
  * GNN (Graph-based spatial relationships)
* ⚡ Cross-Modal Attention for model fusion
* 🔄 Bayesian Updating for continuous learning
* 🔍 Explainable AI using SHAP & LIME
* 🌆 Smart city AQI forecasting & hotspot detection

---

## 🏗️ System Architecture

The system consists of:

1. Data Collection & Preprocessing
2. Environmental Simulation

   * SimPy (Industrial emissions)
   * SUMO (Traffic emissions)
3. Multi-Model Learning (CNN, LSTM, GRU, GNN)
4. Feature Fusion with Attention
5. Bayesian Adaptive Learning
6. Explainability Layer

---

## 📊 Dataset

* Source: Central Pollution Control Board (CPCB), Government of India – AQI dataset collected via API
* Cities: 26 Indian cities
* Time Period: 2019 – 2023
* Features:

  * PM2.5, PM10
  * NO₂, CO, O₃, SO₂

---

## 📈 Results

| Model                 | RMSE      | MAE      | R²        |
| --------------------- | --------- | -------- | --------- |
| CNN                   | 22.14     | 15.72    | 0.847     |
| LSTM                  | 17.83     | 12.11    | 0.891     |
| GRU                   | 16.92     | 11.34    | 0.902     |
| GNN                   | 15.88     | 10.40    | 0.913     |
| **Hybrid (Proposed)** | **13.47** | **9.12** | **0.941** |

---

## 🛠️ Tech Stack

* Python 3.11
* PyTorch & PyTorch Geometric
* SimPy (Simulation)
* SUMO (Traffic modeling)
* Scikit-learn
* SHAP & LIME

---

## 📂 Project Structure

```
aqi-digital-twin-prediction/
│
├── data/
│   ├── raw/                  # Original datasets (CPCB, etc.)
│   ├── processed/            # Cleaned & preprocessed data
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── eda_analysis.ipynb
│   ├── model_training.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── simulation.py         # SimPy + SUMO logic
│   ├── models/
│   │   ├── cnn.py
│   │   ├── lstm.py
│   │   ├── gru.py
│   │   ├── gnn.py
│   ├── fusion.py            # cross-modal attention
│   ├── bayesian_update.py
│   ├── explainability.py    # SHAP, LIME
│
├── results/
│   ├── graphs/
│   ├── metrics.txt
│             
│
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

```bash
git clone https://github.com/your-username/aqi-digital-twin-prediction.git
cd aqi-digital-twin-prediction

pip install -r requirements.txt
python src/run_all.py
```

---

## 🎯 Applications

* Smart city pollution monitoring
* AQI forecasting & alerts
* Policy simulation (traffic, industrial control)
* Environmental decision support

---

## 🔮 Future Work

* IoT integration for real-time data
* Transformer-based models
* Federated learning across cities
* Satellite data integration

---

## 👩‍💻 Authors

* PARHAVI G.V
* N LAHARI
* AMRUTHA VARSHINI P
* VEEKSHITHA P
