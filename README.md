<div align="center">

# ⚡ Smart Electric Grid

### 🔮 Short-Term Power Demand Forecasting Using ARIMA, MLP & ANFIS

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()
[![MAPE](https://img.shields.io/badge/Best%20MAPE-5.34%25-brightgreen.svg)]()

*Predicting electricity consumption for the next hour and next day using machine learning*

[📊 View Results](#-results--model-comparison) • [🚀 Quick Start](#-quick-start) • [📖 Documentation](PROJECT_DOCUMENTATION.md)

</div>

---

## 🎯 Project Overview

The **Smart Electric Grid** project forecasts electricity usage for the **next hour** and **next day** using historical substation data from Telangana, India. This enables utilities to:

- ⚡ **Optimize power generation** - Match supply with demand
- 💰 **Reduce costs** - Avoid over-generation waste  
- 🛡️ **Prevent blackouts** - Anticipate demand spikes
- 🌱 **Improve efficiency** - Better grid management

> **Winner: ANFIS with 5.34% MAPE** — Considered excellent for electricity forecasting!

---

## 📊 Results & Model Comparison

### 🏆 Hourly Prediction Performance

| Model | RMSE (kW) | MAE (kW) | MAPE (%) | R² Score |
|:---:|:---:|:---:|:---:|:---:|
| **ANFIS** 🥇 | **149.27** | **105.13** | **5.34** | **0.9860** |
| MLP 🥈 | 155.00 | 108.00 | 5.47 | 0.9850 |
| ARIMA 🥉 | 206.00 | 149.00 | 7.53 | 0.9750 |

### 📅 Daily Prediction Performance

| Model | RMSE (kW) | MAE (kW) | MAPE (%) |
|:---:|:---:|:---:|:---:|
| **ANFIS** 🥇 | **119.38** | **94.57** | **4.67** |
| MLP 🥈 | 123.00 | 96.00 | 4.74 |
| ARIMA 🥉 | 158.00 | 129.00 | 6.38 |

> 📈 **ANFIS reduces error by ~30% compared to ARIMA!**

---

## 🧠 Models Implemented

### 1️⃣ ARIMA (AutoRegressive Integrated Moving Average)
```
Statistical baseline model using past values and errors
• Configuration: ARIMA(1,1,1)
• Best for: Linear temporal patterns
• MAPE: 7.53% (Hourly)
```

### 2️⃣ MLP (Multi-Layer Perceptron)
```
Neural network with hidden layers learning non-linear patterns
• Architecture: 128 → 64 → 32 neurons
• Activation: ReLU
• MAPE: 5.47% (Hourly)
```

### 3️⃣ ANFIS (Adaptive Neuro-Fuzzy Inference System) 🏆
```
Hybrid fuzzy-neural system with 256 interpretable rules
• Combines fuzzy logic with neural learning
• Provides interpretable IF-THEN rules
• MAPE: 5.34% (Hourly) ← WINNER!
```

---

## 📁 Repository Structure

```
Smart-Electric-Grid/
├── 📓 SmartElectricGrid.ipynb      # Main notebook: preprocessing, modeling, evaluation
├── 📊 Dataset.csv                   # 8,762 hourly records (full year 2021)
├── 📖 PROJECT_DOCUMENTATION.md      # Complete technical documentation
├── 📋 Quick_Reference_Cheat_Sheet.md # Quick guide & model comparison
├── 🎨 Visual_Architecture_Guide.md  # System & model architecture diagrams
├── 📽️ SmartElectricGrid.pptx        # Presentation slides
├── 📜 LICENSE                       # MIT License
└── 📄 README.md                     # You are here!
```

---

## 📈 Dataset Details

| Property | Value |
|:---|:---|
| **Source** | Telangana, India |
| **Records** | 8,762 hourly measurements |
| **Time Span** | Full year 2021 |
| **Target Variable** | POWER (kW) |
| **Features** | Voltage, Current, Power Factor, Temperature, Humidity, Time |
| **Data Split** | 70% Train / 15% Validation / 15% Test |

### 🔧 Engineered Features
- ⏰ **Temporal**: Hour, Day of Week, Month, Weekend flag
- 🔄 **Cyclical Encoding**: hour_sin, hour_cos, month_sin, month_cos
- 📊 **Lag Features**: load_lag_1h, load_lag_24h, load_lag_168h
- 📉 **Rolling Statistics**: 24h rolling mean & standard deviation

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels scipy
```

### Run the Project
```bash
# Clone the repository
git clone https://github.com/shubhranshu-p/Smart-Electric-Grid.git
cd Smart-Electric-Grid

# Open Jupyter Notebook
jupyter notebook SmartElectricGrid.ipynb
```

### Or Run on Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shubhranshu-p/Smart-Electric-Grid/blob/main/SmartElectricGrid.ipynb)

---

## 🔑 Key Findings

### 📊 Top Predictors
1. **Current consumption** — Direct indicator of load
2. **Load 1 hour ago** — Recent past strongly predicts future
3. **Hour of day** — Clear daily patterns exist

### ⏰ Error Patterns
- 🌙 **Nighttime**: Lower errors (stable consumption)
- ☀️ **Daytime (9AM-5PM)**: Higher errors (variable activity)
- 📅 **Wednesday**: Peak weekly error (mid-week surge)
- 🏖️ **Saturday**: Lowest error (predictable weekend)

---

## ⭐ Team Members

<div align="center">

| | | |
|:---:|:---:|:---:|
| ⭐ **Shubhranshu Sudeepta Panda** | ⭐ **Rupesh Kumar Mund** | ⭐ **Akshit Verma** |

</div>

---

## 🔮 Future Improvements

- 🔮 **Implement multi-step forecasting** (e.g., next 24 hours)
- 🌦️ **Integrate weather and external factors**
- ⚡ **Deploy as a real-time forecasting API**
- 🤖 **Try advanced models**: LSTM, GRU, Transformers
- 🎯 **Apply hyperparameter tuning** techniques like Grid Search or Optuna

---

## 📚 Documentation

| Document | Description |
|:---|:---|
| [📖 PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) | Complete technical deep-dive |
| [📋 Quick_Reference_Cheat_Sheet.md](Quick_Reference_Cheat_Sheet.md) | Quick guide for presentations |
| [🎨 Visual_Architecture_Guide.md](Visual_Architecture_Guide.md) | Architecture diagrams |

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

### ⭐ Star this repo if you found it helpful!

**Made with ❤️ for Smart Grid Innovation**

[![GitHub stars](https://img.shields.io/github/stars/shubhranshu-p/Smart-Electric-Grid?style=social)](https://github.com/shubhranshu-p/Smart-Electric-Grid/stargazers)

</div>
