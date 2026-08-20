# 📈 Enterprise Retail Demand Forecasting (PyTorch LSTM)

A robust, deep learning-based time-series forecasting pipeline built in **PyTorch** to predict daily retail sales. This project utilizes a single, unified Long Short-Term Memory (LSTM) network to simultaneously forecast demand across an entire enterprise (10 distinct stores and 50 unique products).

## 🚀 Project Overview

Accurate demand forecasting is the backbone of efficient supply chain management. Traditional statistical models (like ARIMA) often struggle when predicting thousands of store-item combinations simultaneously and fail to capture complex, non-linear cross-product trends. 

This project solves that by engineering temporal features and training a global neural network on **913,000+ historical sales records**, creating a model that learns shared seasonality and demand patterns across the entire business.

### 🛠️ Tech Stack
* **Deep Learning:** PyTorch, `torch.nn.LSTM`
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning Ops:** Scikit-Learn (`MinMaxScaler`)
* **Visualization:** Matplotlib

## 🧠 Architecture & Feature Engineering

To allow a single network to understand demand at 500 different store-item intersections, robust feature engineering was required:

1. **Temporal Cyclical Encoding:** Raw dates (Year, Month, Day of Week) were transformed using sine and cosine waves. This allows the network to mathematically understand that Sunday is right next to Monday, and December flows seamlessly into January, perfectly capturing weekly and annual seasonality.
2. **Global Context Integration:** Store IDs and Item IDs were fed directly into the model alongside the temporal data.
3. **The Network:** 
    * **Input Size:** 8 Features (Sales, Store ID, Item ID, Day_sin, Day_cos, Month_sin, Month_cos, Year)
    * **Hidden Layers:** 2 stacked LSTM layers with 128 hidden units.
    * **Output Layer:** A fully connected linear layer outputting the forecasted sales for the following day ($t+1$).
    * **Sliding Window:** The model looks back at the previous 30 days of data to make its prediction.

## 📊 Model Comparison & Results

The global LSTM network was evaluated against traditional baseline models. The model was trained on data from 2013 to late 2017, and evaluated on unseen validation data from the end of 2017.

### The Showdown (Store 1, Item 1 Benchmark)
* **Traditional ARIMA Baseline SMAPE:** 28.35%
* **Deep Learning LSTM SMAPE:** 20.96%
*(Lower is better)*

By switching to a deep learning architecture, forecasting error was reduced by nearly 8%. 

### Global Enterprise Accuracy
When evaluated across the entire validation dataset (all 10 stores and 50 items simultaneously), the model achieved highly competitive results:

* **Global Mean Absolute Error (MAE):** 6.32 items off per day
* **Global Symmetric Mean Absolute Percentage Error (SMAPE):** 13.55%
