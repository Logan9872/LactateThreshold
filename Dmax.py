
import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.interpolate import interp1d

# 准备数据的函数，这里使用示例数据
def prepare_data():
    data = {
        "step": [0, 1, 2, 3, 4, 5, 6, 7],
        "length": [0, 3, 3, 3, 3, 3, 3, 2],
        "intensity": [0, 50, 75, 100, 125, 150, 175, 191],
        "lactate": [0.93, 0.98, 1.23, 1.88, 2.8, 4.21, 6.66, 8.64],
        "heart_rate": [96, 114, 134, 154, 170, 182, 193, 198]
    }
    df = pd.DataFrame(data)
    return {"data": [df]}
# ——————————————————————————————————————————————————————————————————————————————————
# 曲线拟合函数

def prepare_fit(data, fit="3rd degree polynomial", include_baseline=False, sport="cycling"):
    if data is None:
        raise ValueError("No data provided. Please include your data in the function.")

    if "intensity" not in data.columns or "lactate" not in data.columns:
        raise ValueError("It looks like you didn't prepare your data. Please call `prepare_data()` before.")

    sport_options = ["cycling", "running", "swimming"]
    if sport not in sport_options:
        raise ValueError("Invalid sport. Choose one of 'cycling', 'running', or 'swimming'.")

    # Adjust intensity values to ensure no zero values
    to_subtract = data['intensity'].iloc[2] - data['intensity'].iloc[1]
    data.loc[data['intensity'] == 0, 'intensity'] = data['intensity'].iloc[1] - to_subtract

    if sport == "cycling" or sport == "running":
        interpolation_factor = 0.1
    elif sport == "swimming":
        interpolation_factor = 0.01

    if include_baseline:
        data_for_modeling = data
    else:
        data_for_modeling = data.iloc[1:]

    intensity_range = np.arange(min(data_for_modeling['intensity']), max(data_for_modeling['intensity']), interpolation_factor)

    if fit == "3rd degree polynomial":
        def polynomial_fit(x, a, b, c, d):
            return a * x**3 + b * x**2 + c * x + d

        params, _ = curve_fit(polynomial_fit, data_for_modeling['intensity'], data_for_modeling['lactate'])
        lactate_estimate = polynomial_fit(intensity_range, *params)

    elif fit == "4th degree polynomial":
        def polynomial_fit(x, a, b, c, d, e):
            return a * x**4 + b * x**3 + c * x**2 + d * x + e

        params, _ = curve_fit(polynomial_fit, data_for_modeling['intensity'], data_for_modeling['lactate'])
        lactate_estimate = polynomial_fit(intensity_range, *params)

    elif fit == "B-spline":
        # Perform B-spline interpolation
        spline = interp1d(data_for_modeling['intensity'], data_for_modeling['lactate'], kind='cubic')
        lactate_estimate = spline(intensity_range)

    elif fit == "Exponential":
        def exponential_fit(x, a, b, c):
            return a + b * np.exp(c * x)

        params, _ = curve_fit(exponential_fit, data_for_modeling['intensity'], data_for_modeling['lactate'])
        lactate_estimate = exponential_fit(intensity_range, *params)

    else:
        raise ValueError("Invalid fit method. Choose one of '3rd degree polynomial', '4th degree polynomial', 'B-spline', or 'Exponential'.")

    result = pd.DataFrame({
        'intensity': intensity_range,
        'lactate_estimate': lactate_estimate
    })

    return result
# ——————————————————————————————————————————————————————————————————————————————————

def retrieve_heart_rate(raw_data, intensity_value):
    # Remove baseline value
    raw_data = raw_data.iloc[1:]

    # Linear model heart rate vs intensity
    model = linregress(raw_data["intensity"], raw_data["heart_rate"])

    # Predict heart rate based on intensity value
    out = model.intercept + model.slope * intensity_value

    return round(float(out), 0)


 # ——————————————————————————————————————————————————————————————————————————————————
def retrieve_lactate(model, intensity_value):
    if 'B-spline' in model:
        intensity_range = model['data_augmented']['intensity']
        lactate_range = model['data_augmented']['.fitted']
        f1 = interp1d(intensity_range, lactate_range, kind='cubic')
        method_intensity = fsolve(lambda x: f1(x) - intensity_value, intensity_range[0])
    else:
        fitted_values = model['model'].fittedvalues
        intensity_values = model['model'].model.exog[:, 1]
        try:
            method_intensity = np.interp(intensity_value, fitted_values, intensity_values)
        except ValueError:
            # Handle cases where lactate curve goes down and then increases again
            start_increasing = np.where(np.diff(fitted_values) < 0)[0][-1]
            fitted_values = fitted_values[start_increasing:]
            intensity_values = intensity_values[start_increasing:]
            method_intensity = np.interp(intensity_value, fitted_values, intensity_values)

    return round(float(max(method_intensity)), 1)


# ——————————————————————————————————————————————————————————————————————————————————

# 辅助函数：计算Dmax方法
def helper_dmax(data_prepared, sport, plot):
    data_prepared_dmax = prepare_fit(data_prepared['data'][0], "3rd degree polynomial", False, sport)
    model_coefficients = data_prepared_dmax['model'][0].params

    data_dmax = data_prepared_dmax['data'][0].iloc[1:]

    diff_lactate = np.diff(np.array(data_dmax['lactate']))
    diff_intensity = np.diff(np.array(data_dmax['intensity']))
    max_intensity = data_dmax['intensity'].max()

    lin_beta = diff_lactate / diff_intensity

    d_max_roots = np.roots(
        [model_coefficients[3], model_coefficients[2] - lin_beta, model_coefficients[1], model_coefficients[0]])
    d_max = np.real(d_max_roots[d_max_roots > 0])
    d_max = d_max[d_max <= max_intensity]

    model_intensity = max(d_max)
    model_lactate = retrieve_lactate(data_prepared_dmax['model'][0], model_intensity)

    if model_lactate > 8:
        model_intensity = min(d_max)
        model_lactate = retrieve_lactate(data_prepared_dmax['model'][0], model_intensity)

    if plot:
        data_plot_line = data_dmax.iloc[[0, -1]]
        fitting = "3rd degree polynomial (default)"
        intensity = model_intensity
        lactate = model_lactate
        data_plot_line = [data_plot_line]
    else:
        data_plot_line = None
        fitting = "3rd degree polynomial (default)"
        intensity = model_intensity
        lactate = model_lactate

    if "heart_rate" in data_prepared['data'][0].columns:
        heart_rate = retrieve_heart_rate(data_prepared['data'][0], intensity)
    else:
        heart_rate = None

    if sport == "cycling":
        intensity = round(intensity, 1)
    elif sport == "running":
        intensity = round(intensity, 2)
    elif sport == "swimming":
        intensity = round(intensity, 3)

    result = {
        "fitting": fitting,
        "intensity": intensity,
        "lactate": lactate,
        "heart_rate": heart_rate,
        "data_plot_line": data_plot_line
    }

    return pd.DataFrame([result])


# 画出乳酸曲线的函数
def plot_lactate(data, method):
    plt.figure(figsize=(8, 6))
    plt.plot(data['intensity'], data['lactate'], marker='o', linestyle='-', label=f'{method} Fit')
    plt.xlabel("Intensity")
    plt.ylabel("Lactate")
    plt.title(f"Lactate Threshold Estimation - {method} Method")
    plt.legend()
    plt.grid(True)
    plt.show()


# 调用方法计算乳酸阈值并绘制数据曲线
data_prepared = prepare_data()
result = helper_dmax(data_prepared, "cycling", plot=True)
print(result)

# 绘制乳酸曲线
plot_lactate(result, "Dmax")
