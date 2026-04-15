import pandas as pd
import numpy as np
import logging
import json
import re
from pathlib import Path

# --- CONFIGURATION ---
ARTIFACTS_DIR = Path("artifacts")
LAGS = [1, 2, 4, 12, 24] # Added more lags for better "memory"
ROLL_WINDOWS = [4, 12, 24]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def fix_2400_hour(ts_str):
    """Converts 'DD-MM-YYYY 24:00' to 'DD-MM-YYYY 00:00'."""
    if pd.isna(ts_str) or not isinstance(ts_str, str):
        return ts_str
    if "24:00" in ts_str:
        return ts_str.replace("24:00", "00:00")
    return ts_str

def build_pipeline():
    in_path = ARTIFACTS_DIR / "data_15min.parquet"
    if not in_path.exists():
        logging.error(f"Missing {in_path}. Run Phase 1 first.")
        return

    df = pd.read_parquet(in_path)
    logging.info(f"Loaded {len(df)} rows.")

    # 1. TIMESTAMP REPAIR
    logging.info("Fixing 24:00 timestamp bug...")
    df['timestamp'] = df['timestamp'].apply(fix_2400_hour)
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['timestamp'])

    # 2. AGGRESSIVE PM2.5 TARGET MERGER (Fixed for AIIM/AIIMS)
    logging.info("Executing Regex-based Canonical Mapper for PM2.5...")
    
    # This regex is specifically tuned to catch the AIIM/AIIMS variants we found
    # It catches: PM2.5, PM2_5, PM2 5, PM25, and names like PM2_5_AIIM
    pm25_pattern = r'PM2[._ ]?5|PM_25|PM25'
    pm25_cols = [c for c in df.columns if re.search(pm25_pattern, c, re.I)]
    
    logging.info(f"Target candidates identified: {pm25_cols}")

    # Convert all candidate columns to numeric and clean negatives
    for col in pm25_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].mask(df[col] < 0) 

    # Combine all found columns into one master target
    # bfill(axis=1) takes the first available non-null value across these columns
    df['target_pm25'] = df[pm25_cols].bfill(axis=1).iloc[:, 0]
    
    # 3. STATION-AWARE SORTING (Critical for time-series)
    df = df.sort_values(['station', 'timestamp'])

    # 4. CYCLICAL TIME FEATURES
    # This maps 23:00 and 00:00 to be close together in space
    logging.info("Generating cyclical temporal features...")
    hour = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60.0
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    df['day_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.dayofweek / 7.0)
    df['day_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.dayofweek / 7.0)

    # 5. FEATURE ENGINEERING (Lags & Rolling per station)
    logging.info("Generating station-specific lags and rolling means...")
    new_features = {}
    for n in LAGS:
        new_features[f"pm25_lag_{n}"] = df.groupby('station')['target_pm25'].shift(n)
    for w in ROLL_WINDOWS:
        new_features[f"pm25_roll_{w}"] = df.groupby('station')['target_pm25'].shift(1).transform(lambda x: x.rolling(w).mean())

    df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)

    # 6. DROPPING NULLS (Warm-up period removal)
    initial_count = len(df)
    # We drop rows where target is missing or where lags haven't started yet
    df = df.dropna(subset=['target_pm25', f"pm25_lag_{max(LAGS)}"])
    logging.info(f"Data cleaned. Retained {len(df)} valid rows (Dropped {initial_count - len(df)} rows).")

    # 7. CHRONOLOGICAL SPLIT (70/15/15)
    train_list, val_list, test_list = [], [], []
    for station, group in df.groupby('station'):
        n = len(group)
        if n < 500: # Ensure station has enough data for meaningful training
            logging.warning(f"Station {station} has insufficient data ({n} rows). Skipping.")
            continue
        train_list.append(group.iloc[:int(n*0.7)])
        val_list.append(group.iloc[int(n*0.7):int(n*0.85)])
        test_list.append(group.iloc[int(n*0.85):])

    df_train = pd.concat(train_list)
    df_val = pd.concat(val_list)
    df_test = pd.concat(test_list)

    # 8. STANDARDIZATION (Z-Score Scaling)
    feature_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos'] + list(new_features.keys())
    means = df_train[feature_cols].mean()
    stds = df_train[feature_cols].std().replace(0, 1)

    for d in [df_train, df_val, df_test]:
        d[feature_cols] = (d[feature_cols] - means) / stds

    # 9. EXPORT
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

    logging.info(f"Pipeline Complete. Stations processed: {df['station'].unique()}")

if __name__ == "__main__":
    build_pipeline()