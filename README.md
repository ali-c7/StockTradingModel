# Buy, Sell, Hold - Stock Prediction Model

An interactive Streamlit web application that generates **Buy, Sell, or Hold** trading signals for stocks based on historical price data and technical indicators.

## Features

- 📊 Real-time stock data retrieval from Yahoo Finance
- 📈 Technical indicator analysis (RSI, MACD, Bollinger Bands, Volume MA)
- 🤖 Predictive modeling with both rule-based and ML approaches
- 📉 Interactive visualizations with Plotly
- ✅ Walk-forward validation for robust evaluation

## Tech Stack

- **Framework**: Streamlit
- **Data Source**: Yahoo Finance API (yfinance)
- **Data Processing**: pandas, numpy
- **Technical Indicators**: ta (technical analysis library)
- **Visualization**: Plotly
- **ML**: scikit-learn

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ali-c7/Buy-Sell-Hold-Predictive-Model.git
cd Buy-Sell-Hold-Predictive-Model
```

2. Create and activate virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## Project Structure

```
alpha.ai/
├── app.py                 # Main Streamlit application entry point
├── ui/                    # UI modules
├── data/                  # Data retrieval and preprocessing
├── core/                  # Core business logic and models
├── plots/                 # Visualization modules
├── tests/                 # Unit and integration tests
├── docs/                  # Documentation and plans
└── requirements.txt       # Python dependencies
```

## Development Status

See [FEATURES_PLAN.md](docs/FEATURES_PLAN.md) for detailed development roadmap.

**Current Phase**: Phase 1 - UI Foundation ✅

## Documentation

- [Product Brief](docs/buy_sell_hold_product_brief.md)
- [Features Plan](docs/FEATURES_PLAN.md)
- [Technical Plans](docs/features/)

## Disclaimer

**This tool is for educational and informational purposes only. Not financial advice.**

Always conduct your own research and consult with a qualified financial advisor before making investment decisions.

## License

MIT License - see LICENSE file for details

## Author

Ali Chaudhry

