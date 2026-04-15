# Raipur Air Pollution — PM2.5 Integrated Regression Framework

> A 3-phase machine learning pipeline for predicting PM2.5 concentrations across four Continuous Ambient Air Quality Monitoring Stations (CAAQMS) in Raipur, India.

---

## Project Overview

This project builds an **Integrated Regression Framework** that ingests raw DCR (Daily Continuous Recording) data from four monitoring stations, engineers time-series features, and trains a stacking ensemble that outperforms all individual base learners.

| Station | Location |
|---------|----------|
| AIIMS | All India Institute of Medical Sciences, Raipur |
| BHATAGAON | Bhatagaon, Raipur |
| IGKV | Indira Gandhi Krishi Vishwavidyalaya, Raipur |
| SILTARA | Silatara Industrial Area, Raipur |

Raw data spans **2022–2026**, covering pollutants: NO, NO₂, NOₓ, SO₂, O₃, PM10, PM2.5, Benzene, CO, NH₃, and meteorological parameters (wind speed/direction, temperature, humidity, solar radiation, rainfall).

---

## Pipeline

### Phase 1 — Data Ingestion (`1_combine_raw_csvs.py`)

Handles the messy reality of hundreds of raw CSV files exported from monitoring instruments.

- **Dynamic header detection** — scans each file line-by-line for `"Date & Time"` to skip variable-length metadata blocks at the top of each file
- **Phantom column removal** — drops any column whose name starts with `Unnamed`, caused by trailing commas in the instrument export format
- **File filtering** — skips files whose names contain `MUX`, `POWER OFF`, `CALIBRATION`, or `MONTHLY` (non-data sheets)
- **Row cleaning** — removes the units row (first row after header, e.g. `ug/m3`), summary rows containing `Min`, `Max`, or `Total`, and fully empty rows
- **Metadata tagging** — adds `station`, `source_file`, and `source_sheet` columns to every row
- **Memory-safe concat** — casts each file's DataFrame to `str` before appending to avoid PyArrow mixed-type errors on the 13,896-file merge

**Output:** `artifacts/data_15min.parquet` — **872,208 rows × 139 columns**

---

### Phase 2 — Feature Engineering (`2_preprocess_and_features.py`)

Transforms the raw merged data into a clean, model-ready feature matrix.

**Timestamp repair**
- `fix_2400_hour()` converts midnight timestamps written as `DD-MM-YYYY 24:00` to `DD-MM-YYYY 00:00` before parsing — a common instrument export quirk that breaks `pd.to_datetime`

**PM2.5 target unification**
- Uses a regex pattern (`PM2[._ ]?5|PM_25|PM25`) to identify all PM2.5 column variants across stations (e.g. `PM2.5__BHATAGAON`, `PM2_5_AIIM`, `PM2_5_SILTARA`)
- Merges them into a single `target_pm25` column using `bfill(axis=1)` — takes the first non-null value across all candidate columns per row
- Clips negative readings to `NaN`

**Cyclical time encoding**
- Encodes `hour` (with fractional minutes) and `day_of_week` as sine/cosine pairs so the model understands temporal continuity — hour 23 and hour 0 are adjacent in this space, not 23 units apart

```
hour_sin = sin(2π × hour / 24)     hour_cos = cos(2π × hour / 24)
day_sin  = sin(2π × dow  / 7)      day_cos  = cos(2π × dow  / 7)
```

**Lag & rolling features** (computed per-station to prevent cross-station leakage)

| Feature | Description |
|---------|-------------|
| `pm25_lag_1/2/4/12/24` | PM2.5 value 1, 2, 4, 12, 24 steps back |
| `pm25_roll_4/12/24` | Rolling mean over 4, 12, 24 steps (shifted by 1 to avoid leakage) |

**12 total features** — all confirmed in `artifacts/model_config.json`

**Chronological split (70 / 15 / 15)** — performed per-station to preserve temporal order. Stations with fewer than 500 rows are skipped.

**Z-score standardisation** — mean and std computed on training set only, applied to all splits. Saved to `model_config.json` for reproducibility.

**Output:** `features_train.parquet`, `features_val.parquet`, `features_test.parquet`, `model_config.json`

---

### Phase 3 — Stacking Ensemble (`3_train_and_evaluate.py`)

**Hyperparameter tuning** with `RandomizedSearchCV` + `TimeSeriesSplit(n_splits=5)` — respects temporal ordering during cross-validation.

| Model | Search Space |
|-------|-------------|
| XGBoost | `n_estimators` ∈ {500,1000}, `learning_rate` ∈ {0.01,0.05,0.1}, `max_depth` ∈ {4,6,8}, `subsample`, `colsample_bytree`, `gamma` |
| Random Forest | `n_estimators` ∈ {100,200}, `max_depth` ∈ {10,20,None}, `min_samples_split` ∈ {2,5} |
| Linear Regression | Ridge (baseline, no tuning) |

**Stacking** — a `Ridge(alpha=1.0)` meta-learner is trained on the concatenated predictions of all three base models. Final PM2.5 predictions are clipped to `≥ 0`.

**Output:** `integrated_model.pkl`, `feature_importance.png`, `final_results_plot.png`

---

## Results

### Performance on Held-Out Test Set

| Model | R² Score |
|-------|:--------:|
| Linear Regression | 0.5440 |
| Random Forest | 0.5355 |
| XGBoost | 0.4749 |
| **Integrated Model (Stacking)** | **0.5974** |

The stacking ensemble achieves a **+5.3 pp improvement** over the best individual model, demonstrating that combining diverse learners captures variance that no single model can.

### Results Gallery

![Final Results Plot](artifacts/final_results_plot.png)

---

## Repository Structure

```
.
├── 1_combine_raw_csvs.py          # Phase 1: Raw CSV ingestion & merge
├── 2_preprocess_and_features.py   # Phase 2: Feature engineering
├── 3_train_and_evaluate.py        # Phase 3: Model training & evaluation
├── convert_to_csv.py              # Utility: Excel -> CSV batch converter
├── artifacts/
│   ├── data_15min.parquet         # Merged raw data (872K rows, 139 cols)
│   ├── features_train.parquet     # Training split (70%)
│   ├── features_val.parquet       # Validation split (15%)
│   ├── features_test.parquet      # Test split (15%)
│   ├── model_config.json          # Feature list, means, stds
│   └── final_results_plot.png     # Performance comparison chart
└── README.md
```

---

## Quick Start

```bash
# 1. Ingest and merge all raw CSVs
python 1_combine_raw_csvs.py

# 2. Engineer features
python 2_preprocess_and_features.py

# 3. Train stacking ensemble and evaluate
python 3_train_and_evaluate.py
```

**Dependencies:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `joblib`, `pyarrow`

---

## Data Sources

Raw DCR data from CAAQMS stations operated under the **Chhattisgarh Environment Conservation Board (CECB)**. Data collected at 15-minute intervals (quarterly-hourly) and hourly resolution, covering June 2022 – February 2026.
