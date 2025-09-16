# Omega Portfolio Engine

A regime-aware, client-customizable portfolio optimization engine that detects market regimes and optimizes portfolios for Global Minimum Variance and Omega ratio objectives.

> **🚀 Quick Start**: Run `make start` and open `http://localhost:8501` to access the web interface!

## Features

- **🌐 Web Interface**: Modern, interactive web application for easy portfolio management
- **📊 Portfolio Optimization**: GMV and Omega ratio optimization with real-time results
- **🔄 Walk-Forward Backtesting**: Comprehensive backtesting with performance analysis
- **🎯 Regime Detection**: HMM, LSTM, and GMM-based market regime detection
- **⚙️ Policy Management**: Create, edit, and apply custom investment policies
- **📈 Results Dashboard**: Centralized view of all optimization and backtest results
- **💰 Risk Management**: CVaR constraints, turnover penalties, and sector limits
- **📊 Data Integration**: Yahoo Finance, CSV upload, and sample data support
- **🎨 Interactive Charts**: Real-time visualizations and performance metrics
- **🔧 API & CLI**: FastAPI backend and command-line interface for advanced users

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd omega

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Configuration

1. Copy the example environment file:
```bash
cp env.example .env
```

2. Set your API keys in `.env` (optional for basic usage):
```bash
FRED_API_KEY=your_fred_api_key_here
QUANDL_API_KEY=your_quandl_api_key_here
```

### 🌐 **Preferred Way: Web Application**

The easiest way to use the Omega Portfolio Engine is through the web interface:

```bash
# One command to start everything
make start
```

This will start both the API server and web UI. Then open your browser to: **http://localhost:8501**

#### Alternative: Manual Startup
```bash
# Terminal 1: Start API server
python -m api.main

# Terminal 2: Start web UI
streamlit run app/ui.py
```

#### Web Interface Features:
- **Portfolio Optimization**: Interactive GMV and Omega ratio optimization
- **Backtesting**: Walk-forward backtesting with performance analysis  
- **Regime Detection**: Market regime detection using HMM, LSTM, or GMM
- **Policy Management**: Create and manage custom investment policies
- **Results Dashboard**: View and analyze all optimization and backtest results
- **Data Sources**: Yahoo Finance integration, CSV upload, or sample data

### Alternative: Command Line

#### Demo
```bash
# Run the complete demo
make demo
```

#### API Only
```bash
# Start just the API server
python -m api.main
```

## 🌐 Web Interface Usage

The easiest way to use the Omega Portfolio Engine is through the web interface at `http://localhost:8501`:

### 1. Portfolio Optimization
- **Data Sources**: Choose from Yahoo Finance, CSV upload, or sample data
- **Policy Selection**: Select from pre-built policies or create custom ones
- **Optimization**: Run GMV or Omega ratio optimization with real-time results
- **Visualization**: Interactive charts showing portfolio allocation and performance

### 2. Backtesting
- **Walk-Forward Analysis**: Comprehensive backtesting with rolling windows
- **Performance Metrics**: Sharpe ratio, max drawdown, VaR, CVaR, and more
- **Performance Charts**: Interactive performance history and cumulative returns
- **Policy Integration**: Apply investment policies to backtesting

### 3. Regime Detection
- **Multiple Methods**: HMM, LSTM, or GMM-based regime detection
- **Feature Engineering**: Automatic feature extraction from market data
- **Visualization**: Regime labels and transition probabilities
- **Integration**: Use regime information in optimization and backtesting

### 4. Policy Management
- **Create Policies**: Build custom investment policies with full control
- **Template Library**: Choose from Conservative, Balanced, and Aggressive templates
- **Edit Policies**: Modify existing policies to match client needs
- **Apply Policies**: Use policies across optimization and backtesting

### 5. Results Dashboard
- **Centralized Results**: View all optimization and backtesting results
- **Filtering**: Filter by type, objective, policy, or date
- **Analysis**: Detailed performance metrics and visualizations
- **Export**: Save results for further analysis

## 🔧 Command Line Usage

For advanced users or programmatic access:

### Basic Portfolio Optimization

```python
from engine import DataLoader, GMVOptimizer, OmegaOptimizer
import pandas as pd

# Load data
loader = DataLoader()
data = loader.load_universe_data(
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    macro_series=['DGS10', 'VIXCLS'],
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# GMV Optimization
gmv_optimizer = GMVOptimizer()
gmv_result = gmv_optimizer.solve_gmv_with_returns(data['prices'].pct_change())

# Omega Optimization
omega_optimizer = OmegaOptimizer()
omega_result = omega_optimizer.solve_omega_with_returns(
    data['prices'].pct_change(),
    theta=0.02
)
```

### Regime Detection

```python
from engine import RegimeDetector, FeatureExtractor

# Extract features
feature_extractor = FeatureExtractor()
features = feature_extractor.extract_regime_features(
    data['prices'].pct_change(),
    data['prices'],
    data['macro']
)

# Detect regimes
regime_detector = RegimeDetector(method="hmm", n_regimes=3)
regime_detector.fit(features)
regime_labels = regime_detector.predict(features)
```

### Walk-Forward Backtesting

```python
from engine import WalkForwardEngine, BacktestSimulator

# Define optimizer function
def optimizer_func(returns):
    gmv_optimizer = GMVOptimizer()
    result = gmv_optimizer.solve_gmv_with_returns(returns)
    return pd.Series(result['weights'], index=returns.columns)

# Run walk-forward backtest
walk_forward_engine = WalkForwardEngine(
    train_months=120,
    test_months=12,
    step_months=1
)

results = walk_forward_engine.run_walk_forward(
    returns=data['prices'].pct_change(),
    optimizer_func=optimizer_func
)
```

## Configuration

### Universe Configuration

Edit `configs/universe.yaml` to define your investable universe:

```yaml
assets:
  us_equity:
    - VTI
    - SPY
    - QQQ
  fixed_income:
    - SHY
    - IEF
    - TLT
  commodities:
    - GLD
    - SLV
```

### Policy Configuration

Create client-specific policies in `configs/policy.examples/`:

```yaml
client_profile:
  name: "Conservative Investor"
  risk_tolerance: "low"

return_requirements:
  minimum_acceptable_return: 0.02
  target_return: 0.04

risk_constraints:
  max_drawdown: 0.10
  max_volatility: 0.08

asset_constraints:
  min_fixed_income: 0.60
  max_fixed_income: 0.80
  min_equity: 0.15
  max_equity: 0.35
```

## API Endpoints

### Portfolio Optimization
- `POST /optimize` - Optimize portfolio weights
- `POST /backtest` - Run walk-forward backtest
- `POST /regime` - Detect market regimes

### Job Management
- `POST /jobs` - Create a new job
- `GET /jobs/{job_id}` - Get job status
- `GET /jobs/{job_id}/result` - Get job result
- `DELETE /jobs/{job_id}` - Delete a job

### System
- `GET /health` - Health check

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=engine --cov-report=html

# Run specific test file
pytest tests/test_optimize_gmv.py
```

## Development

### Code Quality

The project uses several tools to maintain code quality:

- **ruff**: Linting and formatting
- **black**: Code formatting
- **mypy**: Type checking
- **pytest**: Testing

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Type check
mypy engine/

# Run tests
pytest
```

### Pre-commit Hooks

Install pre-commit hooks:

```bash
pre-commit install
```

## 🆕 Recent Updates

### Web Interface Enhancements
- **Results Storage**: All optimization and backtest results are now stored and displayed in a centralized dashboard
- **Policy Management**: Create, edit, and apply custom investment policies with full UI support
- **Navigation Fixes**: Improved navigation between pages with proper state management
- **Data Integration**: Seamless Yahoo Finance data integration with error handling
- **Interactive Charts**: Real-time portfolio allocation and performance visualizations

### Backend Improvements
- **Fallback Logic**: Robust backtesting with fallback to simple optimization when walk-forward fails
- **Error Handling**: Better error messages and logging throughout the system
- **API Stability**: Improved API reliability and response handling
- **Session Management**: Proper session state management for web interface

### Developer Experience
- **Easy Startup**: `make start` command to launch both API and UI with one command
- **Better Documentation**: Updated README with clear usage instructions
- **Code Quality**: Improved error handling and logging throughout

## Project Structure

```
omega/
├── README.md
├── pyproject.toml
├── .pre-commit-config.yaml
├── env.example
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── configs/
│   ├── universe.yaml
│   ├── data.yaml
│   ├── backtest.yaml
│   └── policy.examples/
├── research/
│   ├── 00_data_audit.ipynb
│   ├── 10_gmv_baseline.ipynb
│   ├── 20_omega_objective.ipynb
│   ├── 30_regime_detection.ipynb
│   └── 40_robustness.ipynb
├── engine/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   ├── signals/
│   ├── risk/
│   ├── optimize/
│   ├── backtest/
│   ├── explain/
│   ├── report/
│   └── utils/
├── api/
│   ├── main.py
│   └── schemas.py
├── app/
│   └── ui.py
├── tests/
└── mlruns/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Bridgewater Associates for the All Weather strategy inspiration
- The quant finance community for open-source tools and methodologies
- Contributors and users of the project

## Support

For questions, issues, or contributions, please:

1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed information
4. Join our community discussions

## Roadmap

- [ ] Additional regime detection methods
- [ ] More optimization objectives
- [ ] Enhanced risk models
- [ ] Real-time data integration
- [ ] Cloud deployment options
- [ ] Advanced reporting features
- [ ] Machine learning integration
- [ ] Performance attribution
- [ ] ESG integration
- [ ] Multi-currency support
