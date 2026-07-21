# AI-Driven Air Quality Prediction using Hybrid Digital Twin

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/SimPy-Simulation-yellow?style=for-the-badge" alt="SimPy">
  <img src="https://img.shields.io/badge/SUMO-Traffic-0066CC?style=for-the-badge" alt="SUMO">
  <img src="https://img.shields.io/badge/Explainable%20AI-SHAP%20%2F%20LIME-orange?style=for-the-badge" alt="XAI">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

<p align="center">
  <strong>A Hybrid Digital Twin framework combining environmental simulation with deep learning for accurate AQI prediction in smart cities.</strong>
</p>

---

## Overview

This project presents a **Hybrid Digital Twin (HDT)** framework for predicting the Air Quality Index (AQI) in smart city environments. It integrates physical simulations of industrial and traffic emissions with multi-modal deep learning models, enabling accurate, explainable, and adaptive air quality forecasting across 26 Indian cities.

---

## Key Features

| Feature | Description |
|:--------|:------------|
| **Hybrid Digital Twin** | Combines environmental simulation (SimPy + SUMO) with AI-driven prediction |
| **Multi-Model Learning** | CNN, LSTM, GRU, and GNN for spatial, temporal, and graph-based feature extraction |
| **Cross-Modal Attention** | Fusion layer that learns optimal combinations across model outputs |
| **Bayesian Updating** | Continuous learning mechanism that adapts to distributional shifts |
| **Explainable AI** | SHAP and LIME integration for transparent, interpretable predictions |
| **Hotspot Detection** | Identification of high-pollution zones for targeted intervention |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        DATA COLLECTION                              │
│         CPCB API  →  26 Indian Cities  →  2019–2023                │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    ENVIRONMENTAL SIMULATION                         │
│   ┌──────────────┐              ┌──────────────┐                    │
│   │    SimPy     │              │     SUMO     │                    │
│   │ Industrial   │              │   Traffic    │                    │
│   │ Emissions    │              │  Emissions   │                    │
│   └──────┬───────┘              └──────┬───────┘                    │
│          └────────────┬────────────────┘                            │
└───────────────────────┼─────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     MULTI-MODEL LEARNING                            │
│   ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐                      │
│   │  CNN  │  │  LSTM │  │  GRU  │  │  GNN  │                      │
│   │Spatial│  │Temporal│  │Temporal│  │Graph  │                      │
│   └───┬───┘  └───┬───┘  └───┬───┘  └───┬───┘                      │
│       └──────┬───┴──────────┴──────────┘                            │
└──────────────┼──────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│              CROSS-MODAL ATTENTION FUSION                           │
│                    → Combined Feature Space                          │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  BAYESIAN ADAPTIVE LEARNING                         │
│              → Continuous Model Updating                             │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   EXPLAINABILITY LAYER                               │
│                  SHAP  &  LIME                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Deep Learning Models

| Model | Focus | Architecture |
|:------|:------|:-------------|
| **CNN** | Spatial feature extraction | Convolutional layers over pollutant matrices |
| **LSTM** | Long-term temporal patterns | Recurrent gates over time-series sequences |
| **GRU** | Efficient temporal modeling | Simplified recurrent architecture |
| **GNN** | Spatial relationships between cities | Graph convolutions on city connectivity |

---

## Dataset

**Source:** Central Pollution Control Board (CPCB), Government of India — collected via official API

| Property | Detail |
|:---------|:-------|
| Cities | 26 Indian cities |
| Time Period | 2019 – 2023 |
| Pollutants | PM2.5, PM10, NO₂, CO, O₃, SO₂ |

---

## Results

| Model | RMSE | MAE | R² |
|:------|-----:|----:|---:|
| CNN | 22.14 | 15.72 | 0.847 |
| LSTM | 17.83 | 12.11 | 0.891 |
| GRU | 16.92 | 11.34 | 0.902 |
| GNN | 15.88 | 10.40 | 0.913 |
| **Hybrid (Proposed)** | **13.47** | **9.12** | **0.941** |

The proposed Hybrid Digital Twin achieves **16.4% lower RMSE** and **3.2% higher R²** than the best single model.

---

## Technology Stack

| Layer | Technology |
|:------|:-----------|
| **Language** | Python 3.11 |
| **Deep Learning** | PyTorch, PyTorch Geometric |
| **Simulation** | SimPy (industrial emissions), SUMO (traffic modeling) |
| **ML Utilities** | Scikit-learn, NumPy, Pandas |
| **Explainability** | SHAP, LIME |
| **Visualization** | Matplotlib, Seaborn |

---

## Project Structure

```
aqi-digital-twin-prediction/
├── data/
│   ├── raw/                          # Original CPCB datasets
│   └── processed/                    # Cleaned & preprocessed data
├── notebooks/
│   ├── data_preprocessing.ipynb      # Data cleaning & feature engineering
│   ├── eda_analysis.ipynb            # Exploratory data analysis
│   └── model_training.ipynb          # Model training & evaluation
├── src/
│   ├── preprocessing.py              # Data ingestion & preprocessing
│   ├── simulation.py                 # SimPy + SUMO emission simulation
│   ├── models/
│   │   ├── cnn.py                    # CNN model
│   │   ├── lstm.py                   # LSTM model
│   │   ├── gru.py                    # GRU model
│   │   └── gnn.py                    # Graph Neural Network model
│   ├── fusion.py                     # Cross-modal attention fusion
│   ├── bayesian_update.py            # Bayesian adaptive learning
│   └── explainability.py             # SHAP & LIME integration
├── results/
│   ├── graphs/                       # Output visualizations
│   └── metrics.txt                   # Evaluation metrics
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- CUDA-compatible GPU (recommended)
- SUMO traffic simulator ([installation guide](https://sumo.dlr.de/docs/))

### Installation

```bash
git clone https://github.com/parhavigv/India-AQI-Digital-Twin.git
cd India-AQI-Digital-Twin
pip install -r requirements.txt
```

### Running

```bash
# Full pipeline
python src/run_all.py

# Or run individually
python src/preprocessing.py
python src/simulation.py
python src/models/cnn.py
python src/fusion.py
python src/bayesian_update.py
python src/explainability.py
```

---

## Applications

- **Smart city pollution monitoring** — Real-time AQI dashboards for urban environments
- **AQI forecasting & alerts** — Predictive alerts for hazardous air quality events
- **Policy simulation** — Model impact of traffic restrictions and industrial regulations
- **Environmental decision support** — Data-driven guidance for urban planning

---

## Future Work

- IoT sensor integration for real-time data ingestion
- Transformer-based architectures for long-range temporal dependencies
- Federated learning across cities for privacy-preserving collaboration
- Satellite imagery integration for spatial pollutant mapping

---

## Author

**Parhavi G.V** — [GitHub](https://github.com/parhavigv)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
