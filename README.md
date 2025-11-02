# 📊 Simple Trading Signal Predictor

**Train ML models to predict stock trading signals and visualize predictions on unseen data.**

---

## 🎯 What This Does

This is a **simplified machine learning trading signal predictor** that:

1. **Fetches** historical stock data from Yahoo Finance
2. **Calculates** 50+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
3. **Trains** ML models (XGBoost, Random Forest, or LightGBM)
4. **Predicts** BUY/HOLD/SELL signals
5. **Visualizes** predictions on a price chart, highlighting test data the model has NEVER seen

**Goal:** Test if ML can predict stock movements based on technical indicators.

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
# Clone the repository
git clone https://github.com/ali-c7/Buy-Sell-Hold-Predictive-Model
cd alpha.ai

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the App

```bash
streamlit run app.py
```

### 3. Configure & Analyze

1. Enter a stock ticker (e.g., AAPL, NVDA, TSLA)
2. Select timeframe (6M, 1Y, 2Y, 5Y)
3. Choose ML model (XGBoost, Random Forest, LightGBM)
4. Adjust train/test split (default: 80%)
5. Click "Run Analysis"

---

## 📈 What You'll See

### The Main Chart

**Price chart with predicted signals:**
- **Blue shaded region** = Training data (model learned from this)
- **Green shaded region** = Test data (model NEVER saw this!)
- **Orange line** = Train/test split
- Small faded markers = Training predictions (reference)
- **🎯 Large bold markers** = TEST predictions (what matters!)

### Key Metrics

- **Test Accuracy**: How often the model was correct on unseen data
  - 33% = Random guessing (3 classes)
  - 50%+ = Model is learning patterns ✅
  - 60%+ = Model working well! ✅✅

- **Test Samples**: Number of data points in test set

---

## 🧠 How It Works

### 1. Data Collection
Fetches OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance.

### 2. Feature Engineering
Calculates 50+ technical indicators organized into:
- **Trend**: EMAs, SMAs, ADX, Ichimoku
- **Momentum**: RSI, Stochastic, Williams %R, CCI
- **Volatility**: ATR, Bollinger Bands, Keltner Channels
- **Volume**: OBV, VWAP, Volume ratios
- **Price**: Returns, ranges, gaps

### 3. Label Generation
Creates BUY/HOLD/SELL labels based on future price movements:
- **BUY**: Price expected to rise ≥2%
- **HOLD**: Price expected to move <2%
- **SELL**: Price expected to fall ≥2%

### 4. Train/Test Split
Splits data **chronologically** (e.g., 80/20):
- **Training (80%)**: First 80% of data - model learns here
- **Testing (20%)**: Last 20% of data - model tested here
- **Important**: Model NEVER sees test data during training!

### 5. Model Training
Trains one of three ML models:
- **XGBoost**: Gradient boosting (usually most accurate)
- **Random Forest**: Ensemble of decision trees (stable)
- **LightGBM**: Fast gradient boosting (efficient)

### 6. Prediction
Generates predictions for all data, including the unseen test set.

### 7. Visualization
Displays predictions on price chart with clear train/test distinction.

---

## 📁 Project Structure

```
alpha.ai/
├── app.py                     # Main Streamlit app
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
│
├── data/                      # Data fetching
│   └── stock/
│       └── stock_data.py      # Yahoo Finance API
│
├── core/                      # Core functionality
│   ├── features/
│   │   └── technical_features.py  # 50+ indicators
│   ├── labels/
│   │   └── label_generator.py     # BUY/HOLD/SELL labels
│   └── models/
│       └── baseline_models.py     # ML models
│
└── docs/                      # Documentation
    ├── SIMPLIFIED_APP.md      # App usage guide
    └── BEGINNERS_GUIDE.md     # ML trading concepts
```

---

## 🔧 Configuration Options

### Stock Ticker
- Enter any valid stock symbol (e.g., AAPL, GOOGL, MSFT)
- Crypto symbols need suffix (e.g., BTC-USD, ETH-USD)

### Timeframe
- **6M**: 6 months (~126 trading days)
- **1Y**: 1 year (~252 trading days)
- **2Y**: 2 years (~504 trading days) - **Recommended**
- **5Y**: 5 years (~1260 trading days)

### ML Model
- **XGBoost**: Usually most accurate, good default choice
- **Random Forest**: More stable, less prone to overfitting
- **LightGBM**: Fastest, good for experimentation

### Train/Test Split
- **50%**: Half for training, half for testing
- **70%**: More testing data (harder for model)
- **80%**: Balanced (recommended)
- **90%**: More training data (easier for model, less testing)

---

## 📊 Interpreting Results

### Good Result ✅
```
Test Accuracy: 58%
Test region shows:
- BUY signals near price bottoms
- SELL signals near price tops
- Clear pattern recognition
```
→ Model is learning useful patterns!

### Bad Result ❌
```
Test Accuracy: 35%
Test region shows:
- Random BUY/SELL signals
- No correlation with price movements
```
→ Model is just guessing, stock may not be predictable with technical indicators

### What to Try:
- **Different timeframes**: Longer timeframes often work better
- **Different stocks**: Some stocks are more predictable than others
- **Different models**: XGBoost vs Random Forest vs LightGBM
- **Different splits**: Try 70% or 90% train/test

---

## 🎓 Learning Resources

### In-App Documentation:
- **`docs/SIMPLIFIED_APP.md`**: Detailed app usage guide
- **`docs/BEGINNERS_GUIDE.md`**: ML trading concepts explained simply

### Key Concepts:
1. **Train/Test Split**: Why we need unseen data to prove the model works
2. **Technical Indicators**: What RSI, MACD, Bollinger Bands, etc. mean
3. **Machine Learning Models**: How XGBoost, Random Forest, LightGBM work
4. **Accuracy vs Profitability**: Why 55% accuracy can still be profitable

---

## ⚠️ Disclaimer

**This is an educational tool for testing ML prediction capabilities.**

- NOT financial advice
- NOT guaranteed to be profitable
- Past performance ≠ future results
- Use at your own risk

For real trading:
- Paper trade first (simulate without real money)
- Start with small amounts
- Use proper risk management
- Understand the risks

---

## 🛠️ Tech Stack

- **UI**: Streamlit
- **Data**: yfinance (Yahoo Finance API)
- **ML**: scikit-learn, XGBoost, LightGBM
- **Indicators**: ta (Technical Analysis library)
- **Visualization**: Plotly
- **Data Processing**: pandas, numpy

---

## 📝 Requirements

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
yfinance>=0.2.0
ta>=0.10.0
plotly>=5.14.0
scikit-learn>=1.3.0
xgboost>=1.7.0
lightgbm>=3.3.0
joblib>=1.3.0
lxml>=4.9.0
html5lib>=1.1
beautifulsoup4>=4.11.0
```

---

## 🐛 Troubleshooting

### "Not enough data"
- **Cause**: Ticker too new or timeframe too short
- **Fix**: Try longer timeframe or different ticker

### "Module not found"
- **Cause**: Dependencies not installed
- **Fix**: Run `pip install -r requirements.txt`

### "Low test accuracy (~35%)"
- **Cause**: Stock may not be predictable with technical indicators
- **Fix**: Try different stock, timeframe, or model

### Chart not showing signals
- **Cause**: Model predicting mostly HOLD (which doesn't show markers)
- **Fix**: This is okay! HOLD = "no clear signal"

---

## 📞 Support

For questions, issues, or suggestions:
- **GitHub**: [ali-c7/Buy-Sell-Hold-Predictive-Model](https://github.com/ali-c7/Buy-Sell-Hold-Predictive-Model)
- **Issues**: Use GitHub Issues tab

---

## 📜 License

MIT License - Feel free to use, modify, and distribute.

---

**Happy Trading! 📈🤖**
