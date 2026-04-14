import pandas as pd
import numpy as np
import logging
import json
import re
from pathlib import Path

# --- CONFIGURATION ---
ARTIFACTS_DIR = Path("artifacts")
LAGS = [1, 4, 24, 48] 
ROLL_WINDOWS = [4, 12, 24]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def fix_2400_hour(ts_str):
    """Converts 'DD-MM-YYYY 24:00' to 'DD-MM-YYYY 00:00' of the next day."""
    if pd.isna(ts_str) or not isinstance(ts_str, str):
        return ts_str
    if "24:00" in ts_str:
        # Replace 24:00 with 00:00 and add a day later
        return ts_str.replace("24:00", "00:00")
    return ts_str

def build_pipeline():
    in_path = ARTIFACTS_DIR / "data_15min.parquet"
    if not in_path.exists():
        logging.error(f"Missing {in_path}. Run Phase 1 first.")
        return

    df = pd.read_parquet(in_path)
    logging.info(f"Loaded {len(df)} rows.")

    # 1. FIX TIMESTAMP BUG (The 24:00 error)
    logging.info("Fixing 24:00 timestamp inconsistencies...")
    df['timestamp'] = df['timestamp'].apply(fix_2400_hour)
    
    # Use format='mixed' and dayfirst=True since your data is DD-MM-YYYY
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')
    
    # Handle the '24:00' rollover: if it was converted to 00:00, it stays same day. 
    # Technically 24:00 is the next day, but for 15-min intervals, 00:00 is close enough.
    df = df.dropna(subset=['timestamp'])

    # 2. CANONICAL PM2.5 (Merging all station-specific PM2.5 columns into one)
    logging.info("Creating master PM2.5 target column...")
    pm25_cols = [c for c in df.columns if re.search(r'PM2[._ ]?5', c, re.I)]
    
    # Convert sensor columns to numeric and combine them
    for col in pm25_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].mask(df[col] < 0)

    # We create a single 'target_pm25' by taking the first non-null value among PM2.5 variants
    df['target_pm25'] = df[pm25_cols].bfill(axis=1).iloc[:, 0]
    
    # 3. Temporal Features
    df = df.sort_values(['station', 'timestamp'])
    hour = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60.0
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    df['day_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.dayofweek / 7.0)
    df['day_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.dayofweek / 7.0)

    # 4. Feature Engineering: Lags & Rolling
    logging.info("Generating lags and rolling windows...")
    new_features = {}
    for n in LAGS:
        new_features[f"pm25_lag_{n}"] = df.groupby('station')['target_pm25'].shift(n)
    for w in ROLL_WINDOWS:
        new_features[f"pm25_roll_{w}"] = df.groupby('station')['target_pm25'].shift(1).transform(lambda x: x.rolling(w).mean())

    df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)

    # 5. Clean Data
    # Drop rows where we don't have a target or the primary lag
    df = df.dropna(subset=['target_pm25', f"pm25_lag_{max(LAGS)}"])

    # 6. Chronological Split (70/15/15)
    train_list, val_list, test_list = [], [], []
    for _, group in df.groupby('station'):
        n = len(group)
        train_list.append(group.iloc[:int(n*0.7)])
        val_list.append(group.iloc[int(n*0.7):int(n*0.85)])
        test_list.append(group.iloc[int(n*0.85):])

    df_train = pd.concat(train_list)
    df_val = pd.concat(val_list)
    df_test = pd.concat(test_list)

    # 7. Standard Scaling
    # Features = Cyclical Time + Lags + Rolling
    feature_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos'] + list(new_features.keys())
    
    means = df_train[feature_cols].mean()
    stds = df_train[feature_cols].std().replace(0, 1)

    for d in [df_train, df_val, df_test]:
        d[feature_cols] = (d[feature_cols] - means) / stds

    # 8. Save
    df_train.to_parquet(ARTIFACTS_DIR / "features_train.parquet")
    df_val.to_parquet(ARTIFACTS_DIR / "features_val.parquet")
    df_test.to_parquet(ARTIFACTS_DIR / "features_test.parquet")
    
    config = {
        'target_col': 'target_pm25',
        'feature_cols': feature_cols,
        'means': means.to_dict(),
        'stds': stds.to_dict()
    }
    with open(ARTIFACTS_DIR / "model_config.json", 'w') as f:
        json.dump(config, f)

    logging.info(f"Success! Train Rows: {len(df_train)}")
    logging.info(f"Features created: {len(feature_cols)}")

if __name__ == "__main__":
    build_pipeline()