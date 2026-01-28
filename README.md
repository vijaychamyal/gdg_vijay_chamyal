# Capital Pulse - AI Stock Market Intelligence
### GDG AI/ML Inductions 2026 - Problem Statement 1

**Author:** Vijay Chamyal

## Project Overview
Capital Pulse is an AI-driven stock market analysis system built for the GDG AI/ML Inductions 2026. The project addresses Problem Statement 1 by combining quantitative price prediction with qualitative AI explanations.

The system allows users to visualize future price trends and simultaneously understand the reasons behind market movements through an intelligent chatbot. It is divided into two core modules:
1.  **Task 1: Predictive Core** (Time Series Forecasting)
2.  **Task 2: Analytical Chatbot** (RAG-based Explanation)

## Repository Structure
The codebase is organized into the following directory structure:

GDG-Inductions-2026/
├── PS1/
│   ├── Task1/              # Predictive Core (Streamlit Dashboard)
│   │   ├── app.py
│   │   └── requirements.txt
│   │
│   └── Task2/              # Analytical Chatbot (RAG Agent)
│       ├── chatbot.py
│       ├── requirements.txt
│       └── .env            # Environment variables (User must create this)
│
└── README.md               # Main Documentation

## Task 1: Predictive Core (Time Series Forecasting)
**Location:** PS1/Task1/

This module provides a robust dashboard for visualizing historical stock data and generating future price predictions.

### Key Features
* **Dual-Model Support:**
    * **LSTM (Long Short-Term Memory):** A Deep Learning model implemented using PyTorch to capture complex non-linear patterns.
    * **Prophet:** A statistical time series model optimized for seasonality and trend analysis.
* **Forecasting:** Predicts stock prices for the next 7 days.
* **Data Source:** Fetches historical data (2020–2024) directly via Yahoo Finance.
* **Performance Metrics:** Displays real-time accuracy scores including RMSE (Root Mean Square Error), MAE (Mean Absolute Error), and MAPE.
* **Visualization:** Interactive charts built with Plotly and Streamlit.

## Task 2: Analytical Stock Chatbot
**Location:** PS1/Task2/

This module is a natural language chatbot designed to explain market behaviors. It utilizes Retrieval Augmented Generation (RAG) to ground its answers in real-world data.

### Key Features
* **Smart Ticker Identification:** Automatically extracts stock tickers from natural language user queries.
* **RAG Architecture:**
    * **Vector Database:** Uses FAISS for efficient similarity search.
    * **Embeddings:** HuggingFace embeddings for semantic understanding.
    * **LLM:** Integrated with Google Gemini via LangChain for reasoning.
* **Multi-Source Data:** Aggregates recent price data, company fundamentals, and financial news from Finnhub and Yahoo Finance.
* **Contextual Answers:** Capable of answering complex queries such as "Why did Apple stock drop?" or "What caused Tesla's recent movement?"

## Tech Stack
* **Machine Learning:** PyTorch, Prophet, Scikit-learn
* **Large Language Models (LLM):** LangChain, Google Gemini, FAISS Vector DB, HuggingFace Embeddings
* **Data Sources:** Yahoo Finance (yfinance), Finnhub API
* **Frontend & Visualization:** Streamlit, Plotly
* **Language:** Python 


### Setup Instructions

1.  **Clone the Repository:**
    git clone https://github.com/vijaychamyal/gdg_vijay_chamyal.git
    cd GDG-Inductions-2026

2.  **Running Task 1 (Prediction Dashboard):**
    Navigate to the Task 1 directory and install dependencies:
    cd PS1/Task1
    pip install -r requirements.txt

    Run the application:
    streamlit run app.py

3.  **Running Task 2 (Analytical Chatbot):**
    Navigate to the Task 2 directory:
    cd ../Task2

    Install dependencies:
    pip install -r requirements.txt

    Configure API Keys:
    Create a file named .env in the PS1/Task2/ directory and add your API keys:
    GOOGLE_API_KEY=your_gemini_api_key
    FINNHUB_API_KEY=your_finnhub_api_key

    Run the chatbot:
    python chatbot.py
