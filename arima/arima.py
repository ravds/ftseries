import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import itertools
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm

def best_arima(log_ret, p_values=range(5), d_values=range(3), q_values=range(5), forecast_horizon=20, training_size=252):
    pdq = list(itertools.product(p_values, d_values, q_values))
    results_list = []

    best_rmse, best_order, best_model = np.inf, None, None

    # Consider only the most recent training_size points for validation
    training_data = log_ret[-training_size:]

    # Start at the forecast_horizon point
    start_point = forecast_horizon

    for order in tqdm(pdq, desc="Evaluating ARIMA models", unit="model"):
        rmse_list = []

        for i in range(start_point, len(training_data) - forecast_horizon):
            train = training_data[:i]
            test = training_data[i:i+forecast_horizon]
            try:
                model = ARIMA(train, order=order)
                results = model.fit()  # We need something here to capture whatever the last parameters were before we move on

                predictions = results.forecast(steps=forecast_horizon)
                rmse = sqrt(mean_squared_error(test, predictions)) # Could we not use numpy here. Keep this I guess if it's faster, but otherwise numpy probably is better
                rmse_list.append(rmse)

            except Exception as e:
                print(f"Error occurred for order {order}: {e}")
                continue  # Go to next iteration instead of breaking out entirely

        if rmse_list:
            avg_rmse = np.mean(rmse_list)
            results_list.append({'Order': order, 'Avg_RMSE': avg_rmse})

            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                best_order = order
                best_model = results

            print(f"Order: {order}, Average RMSE: {avg_rmse}")

    df_results = pd.DataFrame(results_list)
    return best_rmse, best_order, best_model, df_results