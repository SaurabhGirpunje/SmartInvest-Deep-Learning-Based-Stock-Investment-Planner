# src/finetune_pipeline.py
"""
Finetune pipeline with clear print statements.
"""

import os
from pathlib import Path
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from model_built import build_lstm, build_cnn, build_hybrid


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

DEFAULT_WINDOW = 10
DEFAULT_STEP = 5
DEFAULT_EPOCHS = 30
DEFAULT_BATCH = 16
DEFAULT_FORECAST_DAYS = 5
RANDOM_SEED = 42

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')


# --------------------------
# Utils
# --------------------------
def create_dataset(arr, window=10):
    X, y = [], []
    for i in range(len(arr) - window):
        X.append(arr[i:i + window])
        y.append(arr[i + window])
    return np.array(X), np.array(y)


def directional_accuracy(actual, predicted):
    actual = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()
    if len(actual) < 2:
        return np.nan
    actual_move = actual[1:] - actual[:-1]
    pred_move = predicted[1:] - predicted[:-1]
    return (np.sign(actual_move) == np.sign(pred_move)).mean() * 100


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    eps = 1e-8
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100


def save_prediction_plot(dates, actual, predicted, title, out_path):
    plt.figure(figsize=(10,5))
    plt.plot(dates, actual, label='Actual', linewidth=1.2)
    plt.plot(dates, predicted, label='Predicted', alpha=0.8)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_combined_plot(dates, actual, preds_dict, out_path):
    plt.figure(figsize=(12,6))
    plt.plot(dates, actual, label='Actual', linewidth=1.2)
    for k, v in preds_dict.items():
        plt.plot(dates, v, label=f'{k.upper()} Predicted', alpha=0.8)
    plt.title('All Models Prediction vs Actual (Walk-forward aligned)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_forecast_plot(dates, historical_dates, historical_actual, forecast_dates, forecast_vals, out_path):
    plt.figure(figsize=(10,5))
    plt.plot(historical_dates, historical_actual, label='History', linewidth=1.2)
    plt.plot(forecast_dates, forecast_vals, marker='o', label='Forecast')
    plt.title('5-day Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# --------------------------
# Walk-forward (PRINTS ADDED)
# --------------------------
def walk_forward_eval(prices, dates, model_type='lstm',
                      window=DEFAULT_WINDOW, step=DEFAULT_STEP,
                      epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH,
                      verbose=0):

    print(f"\n--- Walk-forward starting for model: {model_type.upper()} ---")   # NEW PRINT

    preds_all, actuals_all, dates_all = [], [], []

    es = EarlyStopping(monitor='loss', patience=20,
                       restore_best_weights=True, verbose=0)

    loop_count = 0

    for i in range(window, len(prices) - step + 1, step):
        loop_count += 1
        print(f"  Running WF iteration {loop_count}, train={i}, test={i+step}")   # NEW PRINT

        train_prices = prices[:i]
        test_prices = prices[i:i + step]
        train_dates = dates[:i]
        test_dates = dates[i:i + step]

        scaler = StandardScaler()
        tr_scaled = scaler.fit_transform(train_prices.reshape(-1, 1))
        te_scaled = scaler.transform(test_prices.reshape(-1, 1))

        X_train, y_train = create_dataset(tr_scaled, window)
        concat_test = np.vstack([tr_scaled[-window:], te_scaled])
        X_test, y_test = create_dataset(concat_test, window)

        if X_train.size == 0 or X_test.size == 0:
            continue

        X_train = X_train.reshape(-1, window, 1)
        X_test = X_test.reshape(-1, window, 1)

        if model_type == 'lstm':
            model = build_lstm(window)
        elif model_type == 'cnn':
            model = build_cnn(window)
        else:
            model = build_hybrid(window)

        model.fit(X_train, y_train, epochs=epochs,
                  batch_size=batch_size, verbose=verbose,
                  callbacks=[es])
        # print for many epochs are there and which model name is training
        print(f"    Completed training for iteration {loop_count}.")
        print(f"    Model: {model_type.upper()}, Epochs: {len(model.history.history['loss'])}")
        print(f"    Total iterations: {(len(prices) - window) // step}, Completed: {loop_count}")

        pred_scaled = model.predict(X_test, verbose=0).flatten()
        pred_inv = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        actual_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        preds_all.extend(pred_inv)
        actuals_all.extend(actual_inv)
        dates_all.extend(test_dates)

    print(f"--- Walk-forward completed for {model_type.upper()} ---")  # NEW PRINT

    preds_arr = np.array(preds_all)
    actuals_arr = np.array(actuals_all)

    rmse = np.sqrt(mean_squared_error(actuals_arr, preds_arr))

    return {
        'mse': mean_squared_error(actuals_arr, preds_arr),
        'rmse': rmse,
        'mae': mean_absolute_error(actuals_arr, preds_arr),
        'rmse_mean_ratio': rmse / np.mean(actuals_arr),
        'directional_acc': directional_accuracy(actuals_arr, preds_arr),
        'mape': mean_absolute_percentage_error(actuals_arr, preds_arr),
        'preds': preds_arr,
        'actuals': actuals_arr,
        'dates': pd.to_datetime(dates_all)
    }

def write_aggregates(records, output_root="output"):
    df_rows = []
    for r in records:
        for row in r["metrics"]:
            df_rows.append(row)

    new_df = pd.DataFrame(df_rows)

    out_path = Path(output_root) / "all_metrics.csv"

    # If file exists → append without header
    if out_path.exists():
        new_df.to_csv(out_path, mode="a", header=False, index=False)
        print(f"Appended metrics to: {out_path}")
    else:
        new_df.to_csv(out_path, index=False)
        print(f"Saved new metrics file: {out_path}")


# --------------------------
# Retrain + Forecast (PRINTS ADDED)
# --------------------------
def retrain_and_forecast(prices, dates, model_type='lstm',
                         window=DEFAULT_WINDOW,
                         forecast_days=DEFAULT_FORECAST_DAYS,
                         epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH,
                         verbose=0):

    print(f"\n--- Retraining full model for {model_type.upper()} ---")  # NEW PRINT

    scaler_full = StandardScaler()
    scaled_full = scaler_full.fit_transform(prices.reshape(-1, 1))

    X_full, y_full = create_dataset(scaled_full, window)
    X_full = X_full.reshape(-1, window, 1)

    if model_type == 'lstm':
        model = build_lstm(window)
    elif model_type == 'cnn':
        model = build_cnn(window)
    else:
        model = build_hybrid(window)

    es = EarlyStopping(monitor='loss', patience=4,
                       restore_best_weights=True, verbose=0)

    model.fit(X_full, y_full, epochs=epochs,
              batch_size=batch_size, verbose=verbose,
              callbacks=[es])
    print(f"--- Retraining completed for {model_type.upper()} ---")  # NEW PRINT

    print(f"--- Forecasting next {forecast_days} days ---")  # NEW PRINT

    temp = scaled_full[-window:].reshape(1, window, 1)
    preds_scaled = []

    for i in range(forecast_days):
        print(f"  Forecast step {i+1}")  # NEW PRINT
        f = model.predict(temp, verbose=0)
        preds_scaled.append(float(f))
        temp = np.append(temp[:, 1:, :], f.reshape(1, 1, 1), axis=1)

    preds = scaler_full.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    future_dates = pd.date_range(start=pd.to_datetime(dates[-1]),
                                 periods=forecast_days + 1, freq='B')[1:]

    return pd.DataFrame({'Date': future_dates,
                         f'{model_type}_forecast': preds}), model, scaler_full


# --------------------------
# Process stock (PRINTS ADDED + STOCK PREFIX ADDED)
# --------------------------
def process_stock(csv_path: str, output_root: str = 'output',
                  window=DEFAULT_WINDOW, step=DEFAULT_STEP,
                  epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH,
                  forecast_days=DEFAULT_FORECAST_DAYS, verbose=0):

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    parts = csv_path.parts
    sector = parts[-3] if len(parts) >= 3 else "Unknown"
    stock = parts[-1].replace(".csv", "")

    print(f"\n\n======================")
    print(f" STARTING STOCK: {stock}  (Sector: {sector})")
    print(f"======================\n")   # NEW PRINT

    df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
    print(f"Loaded CSV for {stock}, rows = {len(df)}")  # NEW PRINT

    if "Close" not in df.columns:
        if "close" in df.columns:
            df.rename(columns={'close': 'Close'}, inplace=True)
        else:
            raise ValueError("CSV must contain Close column")

    df = df[['Close']].dropna()

    prices = df['Close'].values
    dates = pd.to_datetime(df.index).to_numpy()

    out_dir = Path(output_root) / sector / stock
    out_dir.mkdir(parents=True, exist_ok=True)

    models = ['cnn', 'lstm', 'hybrid']
    results = {}
    all_preds_df = None

    for m in models:
        print(f"\n=== Running model: {m.upper()} for stock {stock} ===")  # NEW PRINT

        res = walk_forward_eval(prices, dates, model_type=m,
                                window=window, step=step,
                                epochs=epochs, batch_size=batch_size,
                                verbose=verbose)

        results[m] = res

        df_m = pd.DataFrame({
            "Date": res['dates'],
            f"actual_{m}": res['actuals'],
            f"pred_{m}": res['preds']
        })

        if all_preds_df is None:
            all_preds_df = df_m
        else:
            all_preds_df = pd.merge(all_preds_df, df_m,
                                    on="Date", how="outer")

    all_preds_df = all_preds_df.sort_values("Date")

    actual_cols = [c for c in all_preds_df.columns if c.startswith("actual_")]
    all_preds_df['actual'] = all_preds_df[actual_cols].bfill(axis=1).iloc[:, 0]
    all_preds_df.drop(columns=actual_cols, inplace=True)

    cols = ["Date", "actual"] + sorted([c for c in all_preds_df if c.startswith("pred_")])
    all_preds_df = all_preds_df[cols]

    file_path = out_dir / f"{stock}_actual_vs_pred.csv"   # NEW FILENAME
    all_preds_df.to_csv(file_path, index=False)
    print(f"Saved: {file_path}")  # NEW PRINT

    # === LAST 6 MONTHS ACTUAL VS PREDICTED ===
    six_months_ago = all_preds_df["Date"].max() - pd.DateOffset(months=6)
    df_6m = all_preds_df[all_preds_df["Date"] >= six_months_ago].copy()

    plot_6m_path = out_dir / f"{stock}_pred_vs_actual_last_6_months.png"

    plt.figure(figsize=(12,6))
    plt.plot(df_6m["Date"], df_6m["actual"], label="Actual", linewidth=1.3)

    for col in df_6m.columns:
        if col.startswith("pred_"):
            plt.plot(df_6m["Date"], df_6m[col], label=col, alpha=0.8)

    plt.title(f"{stock} - Predicted vs Actual (Last 6 Months)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_6m_path)
    plt.close()

    print(f"Saved: {plot_6m_path}")


    metrics_list = []
    preds_dict = {}

    for m in models:
        res = results[m]
        preds_dict[m] = all_preds_df[f"pred_{m}"]

        plot_file = out_dir / f"{stock}_{m}_plot.png"   # NEW FILENAME
        save_prediction_plot(
            all_preds_df['Date'],
            all_preds_df['actual'],
            preds_dict[m],
            f"{m.upper()} Prediction vs Actual",
            plot_file
        )
        print(f"Saved: {plot_file}")  # NEW PRINT

        # Forecast
        fc_df, model_obj, scaler_full = retrain_and_forecast(
            prices, dates, model_type=m,
            window=window, forecast_days=forecast_days,
            epochs=epochs, batch_size=batch_size,
            verbose=verbose
        )

        model_file = out_dir / f"{stock}_{m}_model.keras"
        scaler_file = out_dir / f"{stock}_{m}_scaler.pkl"
        model_obj.save(model_file, include_optimizer=False)
        joblib.dump(scaler_full, scaler_file)

        print(f"Saved model: {model_file}")       # NEW PRINT
        print(f"Saved scaler: {scaler_file}")     # NEW PRINT

        # Forecast save
        fc_file = out_dir / f"{stock}_forecast_5d.csv"
        if fc_file.exists():
            old = pd.read_csv(fc_file, parse_dates=['Date'])
            merged = pd.merge(old, fc_df, on="Date", how="outer")
            merged.to_csv(fc_file, index=False)
        else:
            fc_df.to_csv(fc_file, index=False)
        print(f"Saved: {fc_file}")  # NEW PRINT

        # Forecast plot
        fc_plot = out_dir / f"{stock}_{m}_forecast.png"
        save_forecast_plot(
            dates, dates, prices,
            fc_df['Date'], fc_df[f"{m}_forecast"],
            fc_plot
        )
        print(f"Saved: {fc_plot}")  # NEW PRINT

        # Metrics
        metrics_list.append({
            "Sector": sector,
            "Stock": stock,
            "Model": m.upper(),
            "MSE": res['mse'],
            "RMSE": res['rmse'],
            "MAE": res['mae'],
            "RMSE/MEAN": res['rmse_mean_ratio'],
            "DirectionalAcc": res['directional_acc'],
            "MAPE": res['mape']
        })

    # Combined plot
    combined_file = out_dir / f"{stock}_combined_plot.png"
    save_combined_plot(
        all_preds_df['Date'],
        all_preds_df['actual'],
        {k: v.values for k, v in preds_dict.items()},
        combined_file
    )
    print(f"Saved: {combined_file}")  # NEW PRINT

    metrics_df = pd.DataFrame(metrics_list)
    metrics_file = out_dir / f"{stock}_metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Saved: {metrics_file}\n")  # NEW PRINT

    print(f"=== COMPLETED STOCK: {stock} ===\n")  # NEW PRINT

    rmse_map = {m['Model']: m['RMSE'] for m in metrics_list}
    rmse_mean_map = {m['Model']: m['RMSE/MEAN'] for m in metrics_list}

    best_by_rmse = min(rmse_map.items(), key=lambda x: x[1])
    best_by_rmse_mean = min(rmse_mean_map.items(), key=lambda x: x[1])

        # === SAVE BEST MODEL SUMMARY FILE ===
    best_summary_file = Path(output_root) / "best_model_summary.csv"

    summary_row = pd.DataFrame([{
        "Sector": sector,
        "Stock": stock,
        "Best_By_RMSE_Model": best_by_rmse[0],
        "Best_By_RMSE_Value": best_by_rmse[1],
        "Best_By_RMSE_Mean_Model": best_by_rmse_mean[0],
        "Best_By_RMSE_Mean_Value": best_by_rmse_mean[1]
    }])

    # --- Append safely ---
    if best_summary_file.exists():
        summary_row.to_csv(best_summary_file, mode="a", header=False, index=False)
    else:
        summary_row.to_csv(best_summary_file, index=False)

    print(f"Saved: {best_summary_file}")


    return {
        "sector": sector,
        "stock": stock,
        "metrics": metrics_list,
        "best_by_rmse": best_by_rmse,
        "best_by_rmse_mean": best_by_rmse_mean
    }
