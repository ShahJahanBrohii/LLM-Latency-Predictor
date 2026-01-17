# LLM-Latency-Predictor
A predictive modeling framework using Ridge Regression to estimate per-token inference latency in Large Language Models (Mistral-7B) based on prompt-level features and context size.

# LLM Latency: Per-Token Inference Latency Prediction

This repository contains the implementation and research findings for a predictive modeling approach to estimate **per-token inference latency** in Large Language Models (LLMs). Using a lightweight regression-based system, this project forecasts token generation time with high accuracy, enabling proactive scheduling and resource management for real-time applications.

## 🚀 Overview
Large Language Models achieved remarkable capabilities, but their inference speed remains a challenge. This project focuses on:
- **Granular Modeling:** Predicting latency at the per-token level rather than total throughput.
- **Feature Analysis:** Identifying the dominant impact of **Context Size** on generation delays.
- **Model Efficiency:** Using a lightweight **Ridge Regression** model to minimize deployment overhead.

## 📊 Key Results
- **Dominant Feature:** Quantitative evidence confirms that **Context Size** is the primary driver of latency ($O(N)$ with KV Cache, $O(N^2)$ without).
- **Accuracy:** The model achieved a **Mean Absolute Error (MAE) of 34.03 ms** on the test set.
- **Performance:** Validated on **Mistral-7B-Instruct-v0.3** using 8-bit quantization.

## 🛠️ Tech Stack
- **Model:** Mistral-7B-Instruct-v0.3 (via Hugging Face)
- **Quantization:** 8-bit (bitsandbytes)
- **Analysis:** Python, Scikit-learn, Pandas, Matplotlib/Seaborn
- **Inference:** PyTorch

## 📂 Project Structure
- `Fiver.ipynb`: Jupyter Notebook containing data collection, model training, and visualization.
- `llm_Latency.pdf`: The research paper detailing the problem statement, methodology, and analysis.
- `llm_latency_dataset.csv`: The dataset containing 471 logged token generation events.

## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/LLM-Latency-Predictor.git](https://github.com/yourusername/LLM-Latency-Predictor.git)