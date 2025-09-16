"""
Streamlit UI for the Omega Portfolio Engine.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import yfinance as yf
import structlog

# Setup logging
import logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Omega Portfolio Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API base URL
API_BASE_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-message {
        color: #28a745;
        font-weight: bold;
    }
    .error-message {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">Omega Portfolio Engine</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.title("Navigation")
        page = st.selectbox(
            "Select Page",
            ["Portfolio Optimization", "Backtesting", "Regime Detection", "Results", "Settings"]
        )
    
    # Main content
    if page == "Portfolio Optimization":
        portfolio_optimization_page()
    elif page == "Backtesting":
        backtesting_page()
    elif page == "Regime Detection":
        regime_detection_page()
    elif page == "Results":
        results_page()
    elif page == "Settings":
        settings_page()

def portfolio_optimization_page():
    """Portfolio optimization page."""
    
    st.header("Portfolio Optimization")
    st.markdown("Optimize portfolio weights using GMV or Omega ratio objectives.")
    
    # Initialize session state for returns data
    if 'returns_df' not in st.session_state:
        st.session_state.returns_df = None
    
    # Data input
    st.subheader("Data Input")
    
    # Sample data or upload
    data_source = st.radio("Data Source", ["Sample Data", "Upload Data", "Yahoo Finance"])
    
    if data_source == "Sample Data":
        # Generate sample data
        if st.button("Generate Sample Data"):
            np.random.seed(42)
            n_assets = 5
            n_days = 252
            
            # Generate sample returns
            returns = np.random.normal(0.0005, 0.02, (n_days, n_assets))
            asset_names = [f"Asset_{i+1}" for i in range(n_assets)]
            dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")
            
            st.session_state.returns_df = pd.DataFrame(returns, columns=asset_names, index=dates)
            st.success("Sample data generated!")
        
        if st.session_state.returns_df is not None:
            st.write("Sample Returns Data:")
            st.dataframe(st.session_state.returns_df.head())
        
    elif data_source == "Upload Data":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            st.session_state.returns_df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
            st.write("Uploaded Returns Data:")
            st.dataframe(st.session_state.returns_df.head())
        else:
            st.warning("Please upload a CSV file")
            
    elif data_source == "Yahoo Finance":
        # Yahoo Finance input
        col1, col2 = st.columns(2)
        
        with col1:
            tickers = st.text_input("Tickers (comma-separated)", "AAPL,MSFT,GOOGL,AMZN,TSLA")
        
        with col2:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
            end_date = st.date_input("End Date", datetime.now())
        
        if st.button("Download Data"):
            try:
                ticker_list = [t.strip() for t in tickers.split(",")]
                st.session_state.returns_df = download_yahoo_data(ticker_list, start_date, end_date)
                st.write("Downloaded Returns Data:")
                st.dataframe(st.session_state.returns_df.head())
            except Exception as e:
                st.error(f"Error downloading data: {str(e)}")
    
    # Check if we have data
    if st.session_state.returns_df is None:
        st.warning("Please load data first before optimizing.")
        return
    
    # Optimization parameters
    st.subheader("Optimization Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        objective = st.selectbox("Objective", ["gmv", "omega"])
        theta = st.number_input("Omega Theta", value=0.02, min_value=0.0, max_value=0.1, step=0.01)
        long_only = st.checkbox("Long Only", value=True)
    
    with col2:
        min_weight = st.number_input("Min Weight", value=0.0, min_value=0.0, max_value=1.0, step=0.01)
        max_weight = st.number_input("Max Weight", value=1.0, min_value=0.0, max_value=1.0, step=0.01)
        turnover_penalty = st.number_input("Turnover Penalty", value=0.0, min_value=0.0, step=0.001)
    
    # Risk constraints
    st.subheader("Risk Constraints")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cvar_cap = st.number_input("CVaR Cap", value=None, min_value=0.0, max_value=1.0, step=0.01)
        max_volatility = st.number_input("Max Volatility", value=None, min_value=0.0, max_value=1.0, step=0.01)
    
    with col2:
        max_drawdown = st.number_input("Max Drawdown", value=None, min_value=0.0, max_value=1.0, step=0.01)
        max_single_weight = st.number_input("Max Single Weight", value=None, min_value=0.0, max_value=1.0, step=0.01)
    
    # Run optimization
    if st.button("Optimize Portfolio"):
        try:
            # Prepare request
            request_data = {
                "returns": st.session_state.returns_df.values.tolist(),
                "asset_names": st.session_state.returns_df.columns.tolist(),
                "dates": st.session_state.returns_df.index.strftime("%Y-%m-%d").tolist(),
                "objective": objective,
                "theta": theta,
                "bounds": (min_weight, max_weight),
                "long_only": long_only,
                "cvar_cap": cvar_cap,
                "max_volatility": max_volatility,
                "max_drawdown": max_drawdown,
                "max_weight": max_single_weight,
                "turnover_penalty": turnover_penalty
            }
            
            # Call API
            response = requests.post(f"{API_BASE_URL}/optimize", json=request_data)
            
            if response.status_code == 200:
                result = response.json()
                
                # Display results
                st.success("Portfolio optimization completed successfully!")
                
                # Portfolio weights
                st.subheader("Optimal Portfolio Weights")
                weights_df = pd.DataFrame(
                    list(result["weights"].items()),
                    columns=["Asset", "Weight"]
                )
                weights_df["Weight"] = weights_df["Weight"].apply(lambda x: f"{x:.2%}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(weights_df)
                
                with col2:
                    # Pie chart
                    fig = px.pie(weights_df, values="Weight", names="Asset", title="Portfolio Allocation")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Performance metrics
                st.subheader("Optimization Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Objective Value", f"{result['objective_value']:.4f}")
                
                with col2:
                    st.metric("Status", result["status"])
                
                with col3:
                    st.metric("Solve Time", f"{result['solve_time']:.2f}s")
                
                with col4:
                    st.metric("Total Weight", f"{sum(result['weights'].values()):.2%}")
                
            else:
                st.error(f"Optimization failed: {response.text}")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

def backtesting_page():
    """Backtesting page."""
    
    st.header("Portfolio Backtesting")
    st.markdown("Run walk-forward backtests to evaluate portfolio performance.")
    
    # Initialize session state for returns data
    if 'backtest_returns_df' not in st.session_state:
        st.session_state.backtest_returns_df = None
    
    # Data input (similar to optimization page)
    st.subheader("Data Input")
    
    # Sample data or upload
    data_source = st.radio("Data Source", ["Sample Data", "Upload Data", "Yahoo Finance"])
    
    if data_source == "Sample Data":
        # Generate sample data
        if st.button("Generate Sample Data"):
            np.random.seed(42)
            n_assets = 5
            n_days = 1000  # More data for backtesting
            
            # Generate sample returns
            returns = np.random.normal(0.0005, 0.02, (n_days, n_assets))
            asset_names = [f"Asset_{i+1}" for i in range(n_assets)]
            dates = pd.date_range(start="2020-01-01", periods=n_days, freq="D")
            
            st.session_state.backtest_returns_df = pd.DataFrame(returns, columns=asset_names, index=dates)
            st.success("Sample data generated!")
        
        if st.session_state.backtest_returns_df is not None:
            st.write("Sample Returns Data:")
            st.dataframe(st.session_state.backtest_returns_df.head())
        
    elif data_source == "Upload Data":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            st.session_state.backtest_returns_df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
            st.write("Uploaded Returns Data:")
            st.dataframe(st.session_state.backtest_returns_df.head())
        else:
            st.warning("Please upload a CSV file")
            
    elif data_source == "Yahoo Finance":
        # Yahoo Finance input
        col1, col2 = st.columns(2)
        
        with col1:
            tickers = st.text_input("Tickers (comma-separated)", "AAPL,MSFT,GOOGL,AMZN,TSLA")
        
        with col2:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=1000))
            end_date = st.date_input("End Date", datetime.now())
        
        if st.button("Download Data"):
            try:
                ticker_list = [t.strip() for t in tickers.split(",")]
                st.session_state.backtest_returns_df = download_yahoo_data(ticker_list, start_date, end_date)
                st.write("Downloaded Returns Data:")
                st.dataframe(st.session_state.backtest_returns_df.head())
            except Exception as e:
                st.error(f"Error downloading data: {str(e)}")
    
    # Check if we have data
    if st.session_state.backtest_returns_df is None:
        st.warning("Please load data first before running backtest.")
        return
    
    # Backtest parameters
    st.subheader("Backtest Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        objective = st.selectbox("Objective", ["gmv", "omega"])
        theta = st.number_input("Omega Theta", value=0.02, min_value=0.0, max_value=0.1, step=0.01)
        initial_capital = st.number_input("Initial Capital", value=1000000.0, min_value=1000.0, step=10000.0)
    
    with col2:
        train_months = st.number_input("Training Months", value=120, min_value=12, max_value=240, step=12)
        test_months = st.number_input("Test Months", value=12, min_value=1, max_value=60, step=1)
        step_months = st.number_input("Step Months", value=1, min_value=1, max_value=12, step=1)
    
    # Transaction costs
    st.subheader("Transaction Costs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        transaction_costs = st.number_input("Transaction Costs (bps)", value=5.0, min_value=0.0, max_value=100.0, step=0.1)
        slippage = st.number_input("Slippage (bps)", value=2.0, min_value=0.0, max_value=50.0, step=0.1)
    
    with col2:
        rebalance_frequency = st.selectbox("Rebalance Frequency", ["daily", "weekly", "monthly", "quarterly"])
        long_only = st.checkbox("Long Only", value=True)
    
    # Run backtest
    if st.button("Run Backtest"):
        try:
            # Prepare request
            request_data = {
                "returns": st.session_state.backtest_returns_df.values.tolist(),
                "asset_names": st.session_state.backtest_returns_df.columns.tolist(),
                "dates": st.session_state.backtest_returns_df.index.strftime("%Y-%m-%d").tolist(),
                "objective": objective,
                "theta": theta,
                "initial_capital": initial_capital,
                "train_months": int(train_months),
                "test_months": int(test_months),
                "step_months": int(step_months),
                "transaction_costs": transaction_costs / 10000,  # Convert to decimal
                "slippage": slippage / 10000,  # Convert to decimal
                "rebalance_frequency": rebalance_frequency,
                "long_only": long_only
            }
            
            # Call API
            response = requests.post(f"{API_BASE_URL}/backtest", json=request_data)
            
            if response.status_code == 200:
                result = response.json()
                
                # Display results
                st.success("Backtest completed successfully!")
                
                # Performance metrics
                st.subheader("Performance Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Return", f"{result['total_return']:.2%}")
                    st.metric("Annualized Return", f"{result['annualized_return']:.2%}")
                
                with col2:
                    st.metric("Annualized Volatility", f"{result['annualized_volatility']:.2%}")
                    st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}")
                
                with col3:
                    st.metric("Max Drawdown", f"{result['max_drawdown']:.2%}")
                    st.metric("VaR (95%)", f"{result['var_95']:.2%}")
                
                with col4:
                    st.metric("Annual Turnover", f"{result['annual_turnover']:.2%}")
                    st.metric("Cost Drag", f"{result['cost_drag']:.2%}")
                
                # Performance chart
                st.subheader("Performance Chart")
                
                if result["performance_history"]:
                    perf_df = pd.DataFrame(result["performance_history"])
                    perf_df["date"] = pd.to_datetime(perf_df["date"])
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=perf_df["date"],
                        y=perf_df["cumulative_return"],
                        mode="lines",
                        name="Cumulative Return",
                        line=dict(width=2)
                    ))
                    
                    fig.update_layout(
                        title="Cumulative Returns",
                        xaxis_title="Date",
                        yaxis_title="Cumulative Return",
                        hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error(f"Backtest failed: {response.text}")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

def regime_detection_page():
    """Regime detection page."""
    
    st.header("Market Regime Detection")
    st.markdown("Detect market regimes using HMM, LSTM, or GMM methods.")
    
    # Initialize session state for features data
    if 'features_df' not in st.session_state:
        st.session_state.features_df = None
    
    # Data input
    st.subheader("Data Input")
    
    # Sample data or upload
    data_source = st.radio("Data Source", ["Sample Data", "Upload Data", "Yahoo Finance"])
    
    if data_source == "Sample Data":
        # Generate sample data
        if st.button("Generate Sample Data"):
            np.random.seed(42)
            n_features = 5
            n_days = 1000
            
            # Generate sample features
            features = np.random.normal(0, 1, (n_days, n_features))
            feature_names = [f"Feature_{i+1}" for i in range(n_features)]
            dates = pd.date_range(start="2020-01-01", periods=n_days, freq="D")
            
            st.session_state.features_df = pd.DataFrame(features, columns=feature_names, index=dates)
            st.success("Sample data generated!")
        
        if st.session_state.features_df is not None:
            st.write("Sample Features Data:")
            st.dataframe(st.session_state.features_df.head())
        
    elif data_source == "Upload Data":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            st.session_state.features_df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
            st.write("Uploaded Features Data:")
            st.dataframe(st.session_state.features_df.head())
        else:
            st.warning("Please upload a CSV file")
            
    elif data_source == "Yahoo Finance":
        # Yahoo Finance input
        col1, col2 = st.columns(2)
        
        with col1:
            tickers = st.text_input("Tickers (comma-separated)", "AAPL,MSFT,GOOGL,AMZN,TSLA")
        
        with col2:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=1000))
            end_date = st.date_input("End Date", datetime.now())
        
        if st.button("Download Data"):
            try:
                ticker_list = [t.strip() for t in tickers.split(",")]
                returns_df = download_yahoo_data(ticker_list, start_date, end_date)
                
                # Calculate features
                st.session_state.features_df = calculate_features(returns_df)
                st.write("Calculated Features:")
                st.dataframe(st.session_state.features_df.head())
            except Exception as e:
                st.error(f"Error downloading data: {str(e)}")
    
    # Check if we have data
    if st.session_state.features_df is None:
        st.warning("Please load data first before detecting regimes.")
        return
    
    # Model parameters
    st.subheader("Model Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        method = st.selectbox("Method", ["hmm", "lstm", "gmm"])
        n_regimes = st.number_input("Number of Regimes", value=3, min_value=2, max_value=10, step=1)
        random_state = st.number_input("Random State", value=42, min_value=0, step=1)
    
    with col2:
        if method == "hmm":
            n_iter = st.number_input("Number of Iterations", value=100, min_value=10, max_value=1000, step=10)
        elif method == "lstm":
            sequence_length = st.number_input("Sequence Length", value=60, min_value=10, max_value=200, step=10)
            hidden_units = st.number_input("Hidden Units", value=64, min_value=16, max_value=256, step=16)
            epochs = st.number_input("Epochs", value=100, min_value=10, max_value=1000, step=10)
        elif method == "gmm":
            pass  # No additional parameters for GMM
    
    # Run regime detection
    if st.button("Detect Regimes"):
        try:
            # Prepare request
            request_data = {
                "features": st.session_state.features_df.values.tolist(),
                "feature_names": st.session_state.features_df.columns.tolist(),
                "dates": st.session_state.features_df.index.strftime("%Y-%m-%d").tolist(),
                "method": method,
                "n_regimes": int(n_regimes),
                "random_state": int(random_state)
            }
            
            # Add method-specific parameters
            if method == "hmm":
                request_data["n_iter"] = int(n_iter)
            elif method == "lstm":
                request_data["sequence_length"] = int(sequence_length)
                request_data["hidden_units"] = int(hidden_units)
                request_data["epochs"] = int(epochs)
            
            # Call API
            response = requests.post(f"{API_BASE_URL}/regime", json=request_data)
            
            if response.status_code == 200:
                result = response.json()
                
                # Display results
                st.success("Regime detection completed successfully!")
                
                # Regime labels
                st.subheader("Regime Labels")
                
                regime_df = pd.DataFrame({
                    "Date": features_df.index,
                    "Regime": result["regime_labels"]
                })
                
                st.dataframe(regime_df.head(20))
                
                # Regime timeline
                st.subheader("Regime Timeline")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=regime_df["Date"],
                    y=regime_df["Regime"],
                    mode="lines+markers",
                    name="Regime",
                    line=dict(width=2)
                ))
                
                fig.update_layout(
                    title="Regime Timeline",
                    xaxis_title="Date",
                    yaxis_title="Regime",
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Regime characteristics
                st.subheader("Regime Characteristics")
                
                if result["regime_characteristics"]:
                    char_df = pd.DataFrame(result["regime_characteristics"]).T
                    st.dataframe(char_df)
                
            else:
                st.error(f"Regime detection failed: {response.text}")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

def results_page():
    """Results page."""
    
    st.header("Results")
    st.markdown("View and analyze portfolio optimization results.")
    
    # Placeholder for results
    st.info("Results will be displayed here after running optimizations or backtests.")

def settings_page():
    """Settings page."""
    
    st.header("Settings")
    st.markdown("Configure application settings.")
    
    # API settings
    st.subheader("API Settings")
    
    api_url = st.text_input("API Base URL", value=API_BASE_URL)
    
    if st.button("Test API Connection"):
        try:
            response = requests.get(f"{api_url}/health")
            if response.status_code == 200:
                st.success("API connection successful!")
            else:
                st.error("API connection failed!")
        except Exception as e:
            st.error(f"API connection error: {str(e)}")
    
    # Display settings
    st.subheader("Display Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.checkbox("Show Data Tables", value=True)
        st.checkbox("Show Charts", value=True)
    
    with col2:
        st.checkbox("Show Debug Info", value=False)
        st.checkbox("Show Performance Metrics", value=True)

def download_yahoo_data(tickers, start_date, end_date):
    """Download data from Yahoo Finance."""
    
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    if len(tickers) == 1:
        # Single ticker
        returns = data["Close"].pct_change().dropna()
    else:
        # Multiple tickers
        returns = data["Close"].pct_change().dropna()
    
    return returns

def calculate_features(returns_df):
    """Calculate features from returns data."""
    
    features = pd.DataFrame(index=returns_df.index)
    
    # Price features
    features["returns"] = returns_df.mean(axis=1)
    features["volatility"] = returns_df.std(axis=1) * np.sqrt(252)
    
    # Momentum features
    features["momentum_1m"] = features["returns"].rolling(21).sum()
    features["momentum_3m"] = features["returns"].rolling(63).sum()
    features["momentum_6m"] = features["returns"].rolling(126).sum()
    
    # Volatility features
    features["vol_1m"] = features["volatility"].rolling(21).mean()
    features["vol_3m"] = features["volatility"].rolling(63).mean()
    features["vol_6m"] = features["volatility"].rolling(126).mean()
    
    # Correlation features
    features["correlation"] = returns_df.rolling(63).corr().groupby(level=0).mean().mean(axis=1)
    
    return features.dropna()

if __name__ == "__main__":
    main()
