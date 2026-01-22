import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.holtwinters import Holt
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy import stats

def check_normality(residuals):
    """
    Uji Normalitas Shapiro-Wilk pada residual.
    """
    try:
        stat, p_value = stats.shapiro(residuals)
        return {
            'statistic': stat,
            'p_value': p_value,
            'is_normal': p_value > 0.05
        }
    except:
        return {'statistic': 0, 'p_value': 0, 'is_normal': False}

def calculate_residuals(y_true, y_pred):
    return y_true - y_pred

def check_homoscedasticity(residuals, X):
    """
    Uji Heteroskedastisitas menggunakan Breusch-Pagan.
    H0: Varian error konstan (Homoskedastisitas).
    """
    try:
        if isinstance(X, pd.DataFrame):
            X_val = X.values
        else:
            X_val = X
            
        test_result = het_breuschpagan(residuals, X_val)
        return {
            'lm_stat': test_result[0],
            'p_value': test_result[1],
            'is_homoscedastic': test_result[1] > 0.05
        }
    except:
        return {'p_value': 0, 'is_homoscedastic': False}

def check_autocorrelation(residuals):
    """
    Uji Autokorelasi menggunakan Durbin-Watson.
    Nilai 2.0 = tidak ada autokorelasi.
    Kisaran 1.5 - 2.5 biasanya dianggap aman.
    """
    try:
        dw_stat = durbin_watson(residuals)
        return dw_stat
    except:
        return 0

def calculate_vif(X):
    """
    Menghitung VIF untuk Multikolinearitas.
    """
    try:
        vif_data = pd.DataFrame()
        vif_data["Variabel"] = X.columns
        
        X_float = X.astype(float)
        
        vif_data["VIF"] = [variance_inflation_factor(X_float.values, i) 
                           for i in range(len(X_float.columns))]
        return vif_data
    except:
        return pd.DataFrame()

def perform_forecasting(df, time_col, target_col, periods=5):
    """
    Melakukan forecasting menggunakan Holt's Linear Trend.
    Mendukung kolom waktu berupa Angka (Tahun) atau Tanggal (Date).
    """
    try:
        data = df[[time_col, target_col]].copy()
        data = data.dropna()
        
        is_date = False
        freq = None
        
        try:
            data['temp_date'] = pd.to_datetime(data[time_col], dayfirst=True, errors='coerce')
            
            if data['temp_date'].notna().mean() > 0.8:
                is_date = True
                data = data.dropna(subset=['temp_date'])
                data = data.sort_values(by='temp_date')
                data[time_col] = data['temp_date'] 
                
                if len(data) > 3:
                     diffs = data[time_col].diff().dropna()
                     min_diff = diffs.min()
                     if min_diff >= pd.Timedelta(days=360): freq = 'Y' 
                     elif min_diff >= pd.Timedelta(days=28): freq = 'M' 
                     else: freq = 'D' 
            else:
                data[time_col] = pd.to_numeric(data[time_col], errors='coerce')
                data = data.dropna(subset=[time_col])
                data = data.sort_values(by=time_col)
                is_date = False
                
        except:
             data[time_col] = pd.to_numeric(data[time_col], errors='coerce')
             data = data.dropna(subset=[time_col])
             data = data.sort_values(by=time_col)

        if len(data) < 2:
            return None
            
        y = data[target_col].values
        model = Holt(y, initialization_method="estimated").fit()
        
        forecast_values = model.forecast(periods)
        
        if is_date:
            last_date = data[time_col].max()
            use_freq = freq if freq else 'D'
            
            future_dates = pd.date_range(start=last_date, periods=periods+1, freq=use_freq)[1:]
            
            if len(future_dates) != periods:
                 future_dates = [last_date + pd.Timedelta(days=i*30) for i in range(1, periods+1)]
                 
            future_x = future_dates
        else:
            last_val = int(data[time_col].max())
            future_x = list(range(last_val + 1, last_val + 1 + periods))
        
        forecast_df = pd.DataFrame({
            time_col: future_x,
            'Forecast': forecast_values
        })
        
        fitted_values = model.fittedvalues
        mse = mean_squared_error(y, fitted_values)
        
        return {
            'history': data,
            'forecast': forecast_df,
            'model': model,
            'mse': mse
        }
    except Exception as e:
        print(f"Error forecasting: {e}")
        return None

def perform_backpropagation_forecasting(df, time_col, target_col, periods=5):
    """
    Melakukan forecasting menggunakan Backpropagation (Neural Network / MLPRegressor).
    Menggunakan pendekatan lag (sliding window) untuk konversi ke supervised learning.
    """
    try:
        # 1. Persiapan Data (sama dengan Holt untuk konsistensi)
        data = df[[time_col, target_col]].copy()
        data = data.dropna()
        
        try:
            sample_str = data[time_col].astype(str)
            temp_date = pd.to_datetime(sample_str, dayfirst=True, errors='coerce')
            
            is_likely_date = temp_date.notna().mean() > 0.8
            
            if is_likely_date:
                try:
                    if data[time_col].astype(str).str.len().unique()[0] == 4 and pd.to_numeric(data[time_col], errors='coerce').notna().all():
                        is_likely_date = False
                except:
                    pass

            if is_likely_date:
                data['time_numeric'] = (temp_date - temp_date.min()).dt.days
                is_date = True
                last_date = temp_date.max()
                diffs = temp_date.diff().dropna()
                if not diffs.empty:
                    if diffs.min() >= pd.Timedelta(days=360): freq = 'Y'
                    elif diffs.min() >= pd.Timedelta(days=28): freq = 'M'
                    else: freq = 'D'
                else:
                    freq = 'D'
            else:
                data['time_numeric'] = pd.to_numeric(data[time_col], errors='coerce')
                is_date = False
        except:
            data['time_numeric'] = pd.to_numeric(data[time_col], errors='coerce')
            is_date = False

        data = data.dropna(subset=['time_numeric'])
        data = data.sort_values(by='time_numeric')
        
        if len(data) < 4: 
            return None
            
        # 2. Persiapan Fitur (Lagging)
        y = data[target_col].values
        X_lag = []
        y_lag = []
        
        look_back = min(3, len(y) - 1)
        
        for i in range(len(y) - look_back):
            X_lag.append(y[i:(i + look_back)])
            y_lag.append(y[i + look_back])
            
        X_lag = np.array(X_lag)
        y_lag = np.array(y_lag)
        
        # 3. Scaling
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        
        X_scaled = scaler_x.fit_transform(X_lag)
        y_scaled = scaler_y.fit_transform(y_lag.reshape(-1, 1)).flatten()
        
        # 4. Train MLPRegressor (Backpropagation)
        model = MLPRegressor(
            hidden_layer_sizes=(50, 20),
            activation='relu',
            solver='adam',
            max_iter=2000,
            random_state=42
        )
        model.fit(X_scaled, y_scaled)
        
        # 5. Iterative Forecasting
        current_batch = y[-look_back:].reshape(1, -1)
        forecast_values = []
        
        for _ in range(periods):
            current_batch_scaled = scaler_x.transform(current_batch)
            pred_scaled = model.predict(current_batch_scaled)
            pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
            
            forecast_values.append(pred)
            
            current_batch = np.append(current_batch[:, 1:], pred).reshape(1, -1)
            
        # 6. Generate Future X-Axis
        if is_date:
            use_freq = freq if 'freq' in locals() else 'D'
            future_x = pd.date_range(start=last_date, periods=periods+1, freq=use_freq)[1:]
        else:
            last_val = int(data[time_col].max())
            future_x = list(range(last_val + 1, last_val + 1 + periods))
            
        forecast_df = pd.DataFrame({
            time_col: future_x,
            'Forecast': forecast_values
        })
        
        y_train_pred_scaled = model.predict(X_scaled)
        y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
        mse = mean_squared_error(y_lag, y_train_pred)
        
        return {
            'history': data,
            'forecast': forecast_df,
            'model': model,
            'mse': mse
        }
    except Exception as e:
        print(f"Error BP forecasting: {e}")
        return None

def perform_linear_regression(df, x_col, y_col):
    """
    Analisis Regresi Linear Sederhana via Statsmodels.
    """
    if df is None or x_col not in df.columns or y_col not in df.columns:
        return None
    
    data = df[[x_col, y_col]].dropna()
    if len(data) < 2: return None
        
    X = data[[x_col]]
    y = data[y_col]
    
    X_with_const = sm.add_constant(X)
    
    model = sm.OLS(y, X_with_const).fit()
    
    y_pred = model.predict(X_with_const)
    residuals = y - y_pred
    mse = mean_squared_error(y, y_pred)
    
    return {
        'model': model,
        'r2': model.rsquared,
        'adj_r2': model.rsquared_adj,
        'slope': model.params[x_col],
        'intercept': model.params['const'],
        'mse': mse,
        'X': X,
        'y': y,
        'y_pred': y_pred,
        'residuals': residuals,
        'f_value': model.fvalue,
        'f_pvalue': model.f_pvalue,
        't_values': model.tvalues,
        'p_values': model.pvalues,
        'conf_int': model.conf_int()
    }

def perform_multiple_regression(df, x_cols, y_col):
    """
    Analisis Regresi Linear Berganda via Statsmodels.
    """
    if df is None or not x_cols or y_col not in df.columns: return None
        
    all_cols = x_cols + [y_col]
    data = df[all_cols].dropna()
    if len(data) < len(x_cols) + 1: return None
        
    X = data[x_cols]
    y = data[y_col]
    
    X_with_const = sm.add_constant(X)
    
    model = sm.OLS(y, X_with_const).fit()
    
    y_pred = model.predict(X_with_const)
    residuals = y - y_pred
    mse = mean_squared_error(y, y_pred)
    
    return {
        'model': model,
        'r2': model.rsquared,
        'adj_r2': model.rsquared_adj,
        'intercept': model.params.get('const', 0),
        'coefficients': model.params.drop('const', errors='ignore').to_dict(),
        'mse': mse,
        'X': X,
        'X_with_const': X_with_const, 
        'y_actual': y,
        'y_pred': y_pred,
        'residuals': residuals,
        'f_value': model.fvalue,
        'f_pvalue': model.f_pvalue,
        't_values': model.tvalues,
        'p_values': model.pvalues,
        'params': model.params
    }

def get_basic_info(df):
    """
    Mengembalikan informasi dasar tentang DataFrame.
    
    Args:
        df: pandas DataFrame.
        
    Returns:
        dict: Dictionary berisi jumlah baris, kolom, dan info missing values.
    """
    if df is None:
        return {}
        
    info = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": df.isnull().sum().sum(),
        "missing_per_column": df.isnull().sum().to_dict()
    }
    return info

def get_descriptive_stats(df):
    """
    Menghitung statistik deskriptif untuk kolom numerik.
    
    Args:
        df: pandas DataFrame.
        
    Returns:
        pandas.DataFrame: DataFrame berisi statistik deskriptif.
    """
    if df is None:
        return pd.DataFrame()
        
    numeric_df = df.select_dtypes(include=['number'])
    
    if numeric_df.empty:
        return pd.DataFrame()
        
    stats = numeric_df.describe().T
    
    stats = stats.drop(columns=['25%', '50%', '75%'], errors='ignore')
    
    stats['Median'] = numeric_df.median()
    
    stats['Modus'] = numeric_df.mode().iloc[0]
    
    target_columns = ['count', 'mean', 'std', 'min', 'Median', 'Modus', 'max']
    final_cols = [col for col in target_columns if col in stats.columns]
    
    stats = stats[final_cols]
    
    stats.columns = ['Jumlah Data', 'Rata-rata', 'Std Deviasi', 'Minimum', 'Median', 'Modus', 'Maksimum']
    
    return stats



def get_frequency_dist(df, column):
    """
    Menghitung distribusi frekuensi untuk kolom kategorikal.
    
    Args:
        df: pandas DataFrame.
        column: Nama kolom target.
        
    Returns:
        pandas.DataFrame: DataFrame dengan kolom 'Kategori' dan 'Frekuensi'.
    """
    if df is None or column not in df.columns:
        return pd.DataFrame()
        
    freq = df[column].value_counts().reset_index()
    freq.columns = [column, 'Frekuensi']
    return freq

def get_outliers_iqr(df, column):
    """
    Mendeteksi outlier menggunakan metode IQR.
    """
    if df is None or column not in df.columns:
        return pd.DataFrame()
        
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)].copy()
    
    return {
        'data': outliers,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'count': len(outliers)
    }