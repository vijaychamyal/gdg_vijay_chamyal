import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
from datetime import timedelta
from prophet import Prophet
from prophet.plot import plot_plotly

st.set_page_config(page_title="Capital Pulse | AI Stock Forecast", layout="wide")

st.title("Capital Pulse: AI Stock Intelligence")
st.sidebar.header("Configuration")

# Sidebar Model Selection
model_choice = st.sidebar.selectbox("Select Prediction Model", ["LSTM (Deep Learning)", "Prophet (Statistical)"])
ticker = st.sidebar.text_input("Enter Ticker (e.g., AAPL, MSFT)", "AAPL").upper()

# Settings for LSTM
FIXED_EPOCHS = 100      
LOOKBACK = 60           
HIDDEN_SIZE = 128       
LAYERS = 2             

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE, num_layers=LAYERS):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def get_lstm_data(ticker):
    df = yf.download(ticker, start="2020-01-01", end="2024-12-31", progress=False)
    if df.empty:
        return None, None
    # Handle MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(ticker, axis=1, level=1) if ticker in df.columns.levels[1] else df
        
    return df['Close'].values.reshape(-1, 1), df.index

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

#MAIN LOGIC 

if model_choice == "LSTM (Deep Learning)":
    st.subheader(f" LSTM Deep Learning Model Analysis: {ticker}")
    
    if st.sidebar.button("Train & Forecast (LSTM)"):
        st.info(f"Fetching data for {ticker} (2020-2024)...")
        
        # 1. Data Loading
        raw_data, dates = get_lstm_data(ticker)
        if raw_data is None or len(raw_data) < LOOKBACK:
            st.error("Not enough data found or Invalid Ticker.")
            st.stop()
            
        # 2. Preprocessing
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(raw_data)
        X, y = create_sequences(scaled_data, LOOKBACK)
        
        # Split: 80% Train, 20% Test
        train_size = int(len(X) * 0.8)
        X_train = torch.FloatTensor(X[:train_size])
        y_train = torch.FloatTensor(y[:train_size])
        X_test = torch.FloatTensor(X[train_size:])
        y_test = torch.FloatTensor(y[train_size:])
        
        # 3. Training
        model = LSTMModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        criterion = nn.MSELoss()
        
        progress = st.progress(0)
        status = st.empty()
        
        model.train()
        for epoch in range(FIXED_EPOCHS):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            if epoch % 5 == 0:
                progress.progress((epoch+1)/FIXED_EPOCHS)
                status.text(f"Training Epoch {epoch+1}/{FIXED_EPOCHS} | Loss: {loss.item():.5f}")
                
        progress.empty()
        status.success("Training Complete!")

        # 4. Evaluation
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test).numpy()
        
        test_preds_act = scaler.inverse_transform(test_preds)
        y_test_act = scaler.inverse_transform(y_test.numpy())
        
        rmse = np.sqrt(mean_squared_error(y_test_act, test_preds_act))
        mape = np.mean(np.abs((y_test_act - test_preds_act) / y_test_act)) * 100
        accuracy = 100 - mape
        
        # 5. Future Forecasting (Next 7 Days)
        future_preds = []
        curr_seq = torch.FloatTensor(scaled_data[-LOOKBACK:]).unsqueeze(0)
        
        for _ in range(7):
            with torch.no_grad():
                pred = model(curr_seq)
                future_preds.append(pred.item())
                new_val = pred.unsqueeze(1)
                curr_seq = torch.cat((curr_seq[:, 1:, :], new_val), dim=1)
                
        future_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
        future_dates = [dates[-1] + timedelta(days=i) for i in range(1, 8)]

        # Dashboard UI
        c1, c2, c3 = st.columns(3)
        c1.metric("Model Accuracy", f"{accuracy:.2f}%")
        c2.metric("RMSE Error", f"{rmse:.2f}")
        c3.metric("Next Day Prediction", f"{future_prices[0][0]:.2f}")

        # Plotting
        fig = go.Figure()
        plot_start = max(0, len(dates) - 300)
        
        # Historical Data
        fig.add_trace(go.Scatter(x=dates[plot_start:], y=raw_data.flatten()[plot_start:], 
                                name='Actual History', showlegend=False, line=dict(color='cyan', width=2)))
        
        # Validation Data
        test_dates = dates[train_size + LOOKBACK:]
        fig.add_trace(go.Scatter(x=test_dates, y=test_preds_act.flatten(), 
                                name='Model Validation', line=dict(color='orange', width=2)))
        
        # Forecast Data
        fig.add_trace(go.Scatter(x=future_dates, y=future_prices.flatten(), showlegend=False,
                                name='7-Day Forecast', line=dict(color='#FF0055', width=2)))
        
        fig.update_layout(template="plotly_dark", title=f"{ticker} LSTM Forecast", xaxis_title="Date", yaxis_title="Price ($)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Table
        st.subheader("7-Day Forecast Data")
        df_res = pd.DataFrame({
            "Date": [d.strftime('%Y-%m-%d') for d in future_dates], 
            "Predicted Price": [f"${p:.2f}" for p in future_prices.flatten()]
        })
        st.dataframe(df_res)

elif model_choice == "Prophet (Statistical)":
    st.subheader(f" Prophet Statistical Model Analysis: {ticker}")
    
    if st.sidebar.button("Generate Forecast (Prophet)"):
        try:
            st.info(f"Fetching Data for {ticker}...")
            data = yf.download(ticker, start="2020-01-01", end="2024-12-31", auto_adjust=True, progress=False)
            
            if data.empty:
                st.error("No data found.")
            else:
                # Preprocessing for Prophet
                df = data.reset_index()
                
                # Check column structure (Handle MultiIndex)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1) # Flatten if needed logic varies by yfinance version
                    
                # Ensure we have Date and Close
                if 'Date' in df.columns and 'Close' in df.columns:
                    df = df[['Date', 'Close']]
                else:
                    # Fallback for simple index
                    df = df.iloc[:, [0, 4]] 

                df.columns = ['ds', 'y']
                df['ds'] = df['ds'].dt.tz_localize(None)

                # Train/Test Split (Hide last 7 days for validation)
                train_df = df.iloc[:-7]
                test_df = df.iloc[-7:]

                # Model Training
                m = Prophet(daily_seasonality=True)
                m.fit(train_df)

                # Forecast
                future = m.make_future_dataframe(periods=7)
                forecast = m.predict(future)

                # Accuracy Calculation
                forecast_test = forecast.iloc[-7:]['yhat'].values
                actual_test = test_df['y'].values
                
                mape = np.mean(np.abs((actual_test - forecast_test) / actual_test)) * 100
                accuracy = 100 - mape
                mae = mean_absolute_error(actual_test, forecast_test)
                rmse = np.sqrt(mean_squared_error(actual_test, forecast_test))

                # Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Model Accuracy", f"{accuracy:.2f}%")
                col2.metric("MAE (Avg Error)", f"${mae:.2f}")
                col3.metric("RMSE", f"${rmse:.2f}")

                # Visualization
                st.subheader("Forecast vs Actuals")
                fig = plot_plotly(m, forecast)
                fig.add_scatter(x=test_df['ds'], y=test_df['y'], mode='markers', 
                               name='Actual Values (Test)', marker=dict(color='red', size=8))
                st.plotly_chart(fig, use_container_width=True)

                # Detailed Table
                comparison = pd.DataFrame({
                    "Date": test_df['ds'],
                    "Actual": actual_test,
                    "Predicted": forecast_test,
                    "Diff": actual_test - forecast_test
                })
                st.write("### Validation Data (Last 7 Days)")
                st.dataframe(comparison.style.format({"Actual": "{:.2f}", "Predicted": "{:.2f}", "Diff": "{:.2f}"}))

        except Exception as e:
            st.error(f"Error: {e}")