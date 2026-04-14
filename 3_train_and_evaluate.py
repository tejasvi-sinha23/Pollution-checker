import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer  # Added for NaN handling

# --- CONFIGURATION ---
ARTIFACTS_DIR = Path("artifacts")

def load_data():
    with open(ARTIFACTS_DIR / "model_config.json", 'r') as f:
        config = json.load(f)
    
    train = pd.read_parquet(ARTIFACTS_DIR / "features_train.parquet")
    val = pd.read_parquet(ARTIFACTS_DIR / "features_val.parquet")
    test = pd.read_parquet(ARTIFACTS_DIR / "features_test.parquet")
    
    X_train, y_train = train[config['feature_cols']], train[config['target_col']]
    X_val, y_val = val[config['feature_cols']], val[config['target_col']]
    X_test, y_test = test[config['feature_cols']], test[config['target_col']]
    
    return X_train, y_train, X_val, y_val, X_test, y_test, config['target_col']

def evaluate(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} -> RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

def main():
    X_train, y_train, X_val, y_val, X_test, y_test, target_name = load_data()
    
    # 1. HANDLE NaNs (The Fix)
    print("Imputing missing values...")
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)
    
    results = {}

    # 2. Train Individual Models
    print("Training Linear Regression...")
    lr = LinearRegression().fit(X_train, y_train)
    
    print("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42).fit(X_train, y_train)
    
    print("Training XGBoost...")
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42).fit(X_train, y_train)

    # 3. Model Integration (Stacking)
    print("Integrating Models (Stacking)...")
    val_preds = pd.DataFrame({
        'lr': lr.predict(X_val),
        'rf': rf.predict(X_val),
        'xgb': xgb.predict(X_val)
    })
    meta_model = LinearRegression().fit(val_preds, y_val)

    # 4. Final Evaluation on Test Set
    test_preds_indiv = pd.DataFrame({
        'lr': lr.predict(X_test),
        'rf': rf.predict(X_test),
        'xgb': xgb.predict(X_test)
    })
    final_preds = meta_model.predict(test_preds_indiv)

    print("\n" + "="*30)
    print(" FINAL TEST SET PERFORMANCE ")
    print("="*30)
    results['Linear Regression'] = evaluate(y_test, test_preds_indiv['lr'], "Linear Regression")
    results['Random Forest'] = evaluate(y_test, test_preds_indiv['rf'], "Random Forest")
    results['XGBoost'] = evaluate(y_test, test_preds_indiv['xgb'], "XGBoost")
    results['Integrated Model'] = evaluate(y_test, final_preds, "Integrated Model")

    # 5. Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values[:200], label='Actual PM2.5', color='black', alpha=0.6)
    plt.plot(final_preds[:200], label='Integrated Model', color='red', linestyle='--')
    plt.title("PM2.5 Prediction vs Actual (First 200 Test Samples)")
    plt.ylabel("PM2.5 Concentration")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(ARTIFACTS_DIR / "final_results_plot.png")
    print(f"\nFinal chart saved to {ARTIFACTS_DIR / 'final_results_plot.png'}")

if __name__ == "__main__":
    main()