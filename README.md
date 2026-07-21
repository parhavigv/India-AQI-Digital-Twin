<<<<<<< HEAD
# AI-Driven Air Quality Prediction using Hybrid Digital Twin

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/PyG-GraphNeuralNetworks-EE4C2C?style=for-the-badge" alt="PyG">
  <img src="https://img.shields.io/badge/SimPy-Discrete--Event-yellowgreen?style=for-the-badge" alt="SimPy">
  <img src="https://img.shields.io/badge/SUMO-TrafficSimulation-orange?style=for-the-badge" alt="SUMO">
  <img src="https://img.shields.io/badge/SHAP--LIME-ExplainableAI-red?style=for-the-badge" alt="SHAP">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

<p align="center">
  <strong>Hybrid Digital Twin framework combining physics-based environmental simulations with multi-model deep learning for accurate AQI prediction in smart city environments.</strong>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#key-features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#dataset">Dataset</a> •
  <a href="#results">Results</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#project-structure">Structure</a> •
  <a href="#applications">Applications</a> •
  <a href="#future-work">Future Work</a> •
  <a href="#license">License</a>
</p>

---

## Overview

This project presents a **Hybrid Digital Twin (HDT)** framework for predicting the Air Quality Index (AQI) in smart city environments. It combines physics-based environmental simulations with multi-model deep learning to achieve superior prediction accuracy and real-world adaptability.

The framework integrates **SimPy** for industrial emission modeling, **SUMO** for traffic emission simulation, and an ensemble of deep learning architectures — **CNN, LSTM, GRU, and GNN** — fused via cross-modal attention. Bayesian updating enables continuous adaptation to evolving environmental conditions, while SHAP and LIME provide full model interpretability.

---

## Key Features

| Feature | Description |
|:--------|:------------|
| **Hybrid Digital Twin** | Combines physics-based simulation (SimPy + SUMO) with data-driven deep learning |
| **Multi-Model Ensemble** | CNN for spatial features, LSTM/GRU for temporal patterns, GNN for graph-based spatial relationships |
| **Cross-Modal Attention** | Learned fusion mechanism for combining heterogeneous model outputs |
| **Bayesian Adaptive Learning** | Continuous model updating as new observational data arrives |
| **Explainable AI** | SHAP and LIME for transparent, interpretable feature attribution |
| **Smart City Integration** | AQI forecasting, hotspot detection, and policy simulation support |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         DATA COLLECTION & PREPROCESSING                      │
│            CPCB AQI Dataset — 26 Indian Cities (2019–2023)                  │
│         Features: PM2.5, PM10, NO₂, CO, O₃, SO₂                            │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                       ENVIRONMENTAL SIMULATION LAYER                         │
│                                                                              │
│  ┌──────────────────────────┐        ┌──────────────────────────┐            │
│  │      SimPy Engine        │        │     SUMO Traffic Sim     │            │
│  │  Industrial Emissions    │        │    Vehicle Emissions     │            │
│  │  Discrete-event modeling │        │   Microscopic traffic    │            │
│  └────────────┬─────────────┘        └────────────┬─────────────┘            │
│               └──────────────────┬─────────────────┘                         │
└──────────────────────────────────┼───────────────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        MULTI-MODEL LEARNING LAYER                           │
│                                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                     │
│  │   CNN    │  │   LSTM   │  │   GRU    │  │   GNN    │                     │
│  │ Spatial  │  │ Long-term│  │ Short-   │  │  Graph   │                     │
│  │ Features │  │ Temporal │  │  term    │  │ Spatial  │                     │
│  │          │  │ Patterns │  │ Temporal │  │ Relations│                     │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘                     │
│       └──────────────┴─────────────┴──────────────┘                          │
│                              │                                               │
│                              ▼                                               │
│               ┌──────────────────────────┐                                   │
│               │  Cross-Modal Attention    │                                   │
│               │    Feature Fusion         │                                   │
│               └─────────────┬────────────┘                                   │
└─────────────────────────────┼────────────────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        ADAPTIVE LEARNING LAYER                              │
│                                                                              │
│            Bayesian Updating for Continuous Model Adaptation                 │
│            Posterior ← Prior × Likelihood(Observed Data)                     │
└─────────────────────────────┬────────────────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         EXPLAINABILITY LAYER                                │
│                                                                              │
│           SHAP (Shapley Additive Explanations)                              │
│           LIME (Local Interpretable Model-agnostic Explanations)            │
│                                                                              │
│           → Feature contribution rankings                                    │
│           → Per-prediction interpretability                                  │
│           → Policy-relevant insight extraction                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Dataset

| Property | Detail |
|:---------|:-------|
| **Source** | Central Pollution Control Board (CPCB), Government of India |
| **Collection Method** | API-based data retrieval |
| **Cities** | 26 Indian cities |
| **Time Period** | 2019 – 2023 |
| **Target Variable** | Air Quality Index (AQI) |
| **Pollutant Features** | PM2.5, PM10, NO₂, CO, O₃, SO₂ |

---

## Results

| Model | RMSE | MAE | R² |
|:------|-----:|----:|---:|
| CNN | 22.14 | 15.72 | 0.847 |
| LSTM | 17.83 | 12.11 | 0.891 |
| GRU | 16.92 | 11.34 | 0.902 |
| GNN | 15.88 | 10.40 | 0.913 |
| **Hybrid (Proposed)** | **13.47** | **9.12** | **0.941** |

> The proposed hybrid model achieves **39.1% lower RMSE** and **24.7% higher R²** compared to the baseline CNN, demonstrating the effectiveness of multi-model fusion with attention-based integration.

---

## Tech Stack

| Category | Technology |
|:---------|:-----------|
| **Language** | Python 3.11 |
| **Deep Learning** | PyTorch, PyTorch Geometric |
| **Simulation** | SimPy (industrial emissions), SUMO (traffic modeling) |
| **Machine Learning** | Scikit-learn |
| **Explainability** | SHAP, LIME |
| **Data Processing** | pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |

---

## Project Structure
=======
# 🌍 AI-Driven Air Quality Prediction using Hybrid Digital Twin

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
>>>>>>> 91c87cf8622d99c174a8278aca99f9c97fccaafa

```
aqi-digital-twin-prediction/
│
├── data/
<<<<<<< HEAD
│   ├── raw/                        # Original CPCB datasets
│   └── processed/                  # Cleaned & preprocessed data
│
├── notebooks/
│   ├── data_preprocessing.ipynb    # Data cleaning & feature engineering
│   ├── eda_analysis.ipynb          # Exploratory data analysis
│   └── model_training.ipynb        # Model training & evaluation
│
├── src/
│   ├── preprocessing.py            # Data ingestion & transformation
│   ├── simulation.py               # SimPy + SUMO simulation logic
│   ├── models/
│   │   ├── cnn.py                  # Convolutional Neural Network
│   │   ├── lstm.py                 # Long Short-Term Memory
│   │   ├── gru.py                  # Gated Recurrent Unit
│   │   └── gnn.py                  # Graph Neural Network
│   ├── fusion.py                   # Cross-modal attention fusion
│   ├── bayesian_update.py          # Bayesian adaptive learning
│   ├── explainability.py           # SHAP & LIME integration
│   └── run_all.py                  # End-to-end pipeline runner
│
├── results/
│   ├── graphs/                     # Training curves, prediction plots
│   └── metrics.txt                 # Evaluation metrics
=======
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
>>>>>>> 91c87cf8622d99c174a8278aca99f9c97fccaafa
│
├── requirements.txt
└── README.md
```

---

<<<<<<< HEAD
## Getting Started

### Prerequisites

- Python 3.11
- CUDA-compatible GPU (recommended for model training)
- SUMO traffic simulator (for simulation module)

### Installation
=======
## ▶️ How to Run
>>>>>>> 91c87cf8622d99c174a8278aca99f9c97fccaafa

```bash
git clone https://github.com/your-username/aqi-digital-twin-prediction.git
cd aqi-digital-twin-prediction
<<<<<<< HEAD
pip install -r requirements.txt
```

### Running

```bash
=======

pip install -r requirements.txt
>>>>>>> 91c87cf8622d99c174a8278aca99f9c97fccaafa
python src/run_all.py
```

---

<<<<<<< HEAD
## Applications

| Domain | Use Case |
|:-------|:---------|
| **Smart Cities** | Real-time pollution monitoring and heatmap generation |
| **Public Health** | AQI forecasting and early warning alerts |
| **Urban Planning** | Policy simulation for traffic regulation and industrial controls |
| **Environmental Governance** | Data-driven decision support for emission control strategies |

---

## Future Work

- **IoT Integration** — Real-time data ingestion from distributed sensor networks
- **Transformer Models** — Attention-based architectures for long-range temporal dependencies
- **Federated Learning** — Privacy-preserving model training across multiple cities
- **Satellite Data** — Remote sensing inputs for enhanced spatial coverage
- **Edge Deployment** — Lightweight models for on-device inference at monitoring stations

---

## Author

**Parhavi G.V** — [GitHub](https://github.com/parhavigv)

B.Tech — Artificial Intelligence & Machine Learning  
Dayananda Sagar University, Bengaluru

---

*Feel free to open issues or submit pull requests for suggestions and improvements.*

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
=======
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

>>>>>>> 91c87cf8622d99c174a8278aca99f9c97fccaafa
