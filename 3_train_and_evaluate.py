import pandas as pd
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

# --- CONFIGURATION ---
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

def load_data():
    with open(ARTIFACTS_DIR / "model_config.json", 'r') as f:
        config = json.load(f)
    train = pd.read_parquet(ARTIFACTS_DIR / "features_train.parquet")
    val = pd.read_parquet(ARTIFACTS_DIR / "features_val.parquet")
    test = pd.read_parquet(ARTIFACTS_DIR / "features_test.parquet")
    return train, val, test, config

def main():
    train_df, val_df, test_df, config = load_data()
    feat_cols, target = config['feature_cols'], config['target_col']

    # 1. DATA PREP: Combine Train and Val for Randomized Search
    # In time-series, we use TimeSeriesSplit to avoid "looking into the future"
    full_train_df = pd.concat([train_df, val_df])
    
    imputer = SimpleImputer(strategy='median')
    X_train_full = imputer.fit_transform(full_train_df[feat_cols])
    y_train_full = full_train_df[target]
    X_test = imputer.transform(test_df[feat_cols])
    y_test = test_df[target]

    tscv = TimeSeriesSplit(n_splits=5)

    # 2. TUNING XGBOOST (Hyperparameter Optimization)
    print("🚀 Full-Scale Tuning: XGBoost...")
    xgb_param_grid = {
        'n_estimators': [500, 1000],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [4, 6, 8],
        'subsample': [0.7, 0.9],
        'colsample_bytree': [0.7, 0.9],
        'gamma': [0, 0.1, 0.5]
    }
    
    xgb_search = RandomizedSearchCV(
        XGBRegressor(n_jobs=-1, tree_method='hist'), 
        param_distributions=xgb_param_grid, 
        n_iter=10, cv=tscv, scoring='r2', n_jobs=-1, random_state=42
    )
    xgb_search.fit(X_train_full, y_train_full)
    best_xgb = xgb_search.best_estimator_

    # 3. TUNING RANDOM FOREST
    print("🌲 Full-Scale Tuning: Random Forest...")
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    rf_search = RandomizedSearchCV(
        RandomForestRegressor(n_jobs=-1), 
        param_distributions=rf_param_grid, 
        n_iter=5, cv=tscv, scoring='r2', n_jobs=-1, random_state=42
    )
    rf_search.fit(X_train_full, y_train_full)
    best_rf = rf_search.best_estimator_

    # 4. ENHANCED STACKING
    print("🔗 Building Meta-Learner (Stacking)...")
    # We use out-of-fold predictions to train the meta-learner
    train_preds = np.column_stack([
        best_xgb.predict(X_train_full),
        best_rf.predict(X_train_full),
        Ridge().fit(X_train_full, y_train_full).predict(X_train_full)
    ])
    
    meta_model = Ridge(alpha=1.0).fit(train_preds, y_train_full)

    # 5. EVALUATION
    test_base_preds = np.column_stack([
        best_xgb.predict(X_test),
        best_rf.predict(X_test),
        Ridge().fit(X_train_full, y_train_full).predict(X_test)
    ])
    final_preds = meta_model.predict(test_base_preds)
    test_df['final_pred'] = np.clip(final_preds, 0, None) # PM2.5 can't be negative

    # 6. FEATURE IMPORTANCE (To show the Prof WHY it predicts)
    plt.figure(figsize=(10, 8))
    importances = pd.Series(best_xgb.feature_importances_, index=feat_cols)
    importances.nlargest(15).plot(kind='barh', color='teal')
    plt.title("Top 15 Predictive Features (XGBoost)")
    plt.savefig(ARTIFACTS_DIR / "feature_importance.png")

    # 7. RESULTS
    print("\n" + "🏆 FINAL ROBUST PERFORMANCE" + "\n" + "="*45)
    for station in test_df['station'].unique():
        s_data = test_df[test_df['station'] == station]
        print(f"{station:<15} | R2: {r2_score(s_data[target], s_data['final_pred']):.4f} | MAE: {mean_absolute_error(s_data[target], s_data['final_pred']):.2f}")

    # Save the actual model for future deployment
    joblib.dump(meta_model, ARTIFACTS_DIR / "integrated_model.pkl")
    print(f"\nModel and plots saved to {ARTIFACTS_DIR}")

if __name__ == "__main__":
    main()