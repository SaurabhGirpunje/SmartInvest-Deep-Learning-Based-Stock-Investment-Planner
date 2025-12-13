# 📈 SmartInvest - AI-Driven Stock Forecasting & Portfolio Planner

SmartInvest is a **deep learning–based stock investment planning system** that combines **short-term stock price forecasting** with **risk-aware portfolio optimization**.  
The project was developed as a **capstone project** with a focus on learning how AI predictions can be translated into structured portfolio decisions in the Indian stock market.

---

## 🚀 Live Demo (Streamlit App)

👉 **Try the application here:**  
https://dl-stock-investment-planner.streamlit.app/



---

## 🧠 Project Overview

Financial markets are highly volatile and sector-dependent, making both **price forecasting** and **portfolio construction** challenging.  
This project explores how **deep learning time-series models** can be used alongside **classical portfolio optimization techniques** to support data-driven investment planning.

SmartInvest provides an **end-to-end pipeline**:
- Sector-wise stock price forecasting  
- Model comparison and selection  
- Portfolio construction and evaluation  

---

## 🔍 Key Features

### 📊 Plots Mode
- Sector-wise stock selection (NSE)
- **Actual vs Predicted price plots**
- Comparison of **CNN, LSTM, and Hybrid CNN-LSTM** models
- Helps visualize short-term forecasting behavior stock by stock

### 🧩 Portfolio Builder Mode
- Portfolio construction using:
  - **Mean-Variance Portfolio (MVP)**
- Displays a **“Stock Table — Mean Variance Portfolio”**
  - Compares **predicted vs actual portfolio values**
  - Helps assess how forecast quality impacts portfolio outcomes
- Risk-aware allocation and diversification analysis

---

## 🧪 Modeling Approach

The system evaluates multiple deep learning architectures for each stock:
- **CNN** — captures short-term local patterns
- **LSTM** — captures long-term temporal dependencies
- **Hybrid CNN-LSTM** — combines both strengths

Instead of using a single model globally, the **best-performing model is selected per stock** based on **prediction stability and normalized error metrics**.

---

## 📐 Evaluation Strategy

Model performance is assessed using:
- RMSE
- RMSE / Mean (primary model selection metric)
- Directional Accuracy
- MAPE

Portfolio-level evaluation compares **predicted vs actual portfolio values** rather than relying only on model-level metrics.

---

## ⚠️ Important Disclaimer

This project was developed **purely for academic learning and experimentation**.  
We implemented **MVP as foundational portfolio optimization techniques**, while acknowledging that **real-world portfolio construction involves more complex strategies, constraints, and market considerations**.

The system is **not intended as financial or investment advice**.

---

## 👥 Intended Users

SmartInvest is designed to be explored by:
- **Individual investors**
- **Students and learners**
- **Anyone interested in experimenting with data-driven portfolio construction**

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Pandas, NumPy
- Plotly
- Streamlit
- Yahoo Finance (data source)

---

## ⭐ Final Note

This project represents a step toward understanding how **AI, time-series forecasting, and portfolio theory** can be combined into a transparent, end-to-end investment planning system.

