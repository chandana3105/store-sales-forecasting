# Data

Raw data files are **not committed** to this repository (they are excluded by `.gitignore`).

## Download Instructions

### Option A — Kaggle CLI (recommended)
```bash
# Install the Kaggle CLI if you haven't already
pip install kaggle

# Place your kaggle.json API key at ~/.kaggle/kaggle.json
# Download competition data
kaggle competitions download -c store-sales-time-series-forecasting -p data/raw/
cd data/raw && unzip store-sales-time-series-forecasting.zip
```

### Option B — Manual download
1. Go to the [competition data page](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)
2. Download all files and place them in `data/raw/`

## Expected Files

```
data/raw/
├── train.csv             # Historical sales (3M+ rows)
├── test.csv              # Rows to predict
├── stores.csv            # Store metadata (city, state, type, cluster)
├── oil.csv               # Daily oil prices (external economic signal)
├── holidays_events.csv   # Ecuador national/regional/local holidays
└── transactions.csv      # Daily transaction counts per store
```

## File Descriptions

| File | Rows | Key columns |
|---|---|---|
| `train.csv` | 3,000,888 | date, store_nbr, family, sales, onpromotion |
| `test.csv` | 28,512 | date, store_nbr, family, onpromotion |
| `stores.csv` | 54 | store_nbr, city, state, type, cluster |
| `oil.csv` | 1,218 | date, dcoilwtico |
| `holidays_events.csv` | 350 | date, type, locale, locale_name, description |
| `transactions.csv` | 83,488 | date, store_nbr, transactions |
