#########################################################################################################
#######################| РЕАЛИЗАЦИЯ ВСПОМОГАТЕЛЬНЫХ ФУНКЦИЙ QA_QC_kern |##############################
#########################################################################################################
import inspect
import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from sklearn.linear_model import LinearRegression


def sigma_counter(df, how_many_sigmas=1):
    return df['Flat'].mean() - how_many_sigmas * df['Flat'].std(), df['Flat'].mean() + how_many_sigmas * df[
        'Flat'].std()


def linear_regressor(df):
    alf_PS = df.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
    Poro = df.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(alf_PS, Poro)  # perform linear regression

    # The coefficients of linear gerression
    k = linear_regressor.coef_
    b = linear_regressor.intercept_

    return k, b


def bourders_initializer(x, y, inner_limit=1):
    df = pd.DataFrame({'X': x, 'Y': y})
    k = linear_regressor(df)[0]  # k of k*x + b
    b = linear_regressor(df)[1]  # b of k*x + b
    df["Flat"] = pd.Series(np.nan, index=df.index)
    df["Flat"] = df.apply(lambda row: row['Y'] - (k * row['X'] + b), axis=1)
    sigmaDown = sigma_counter(df, inner_limit)[0]
    sigmaUp = sigma_counter(df, inner_limit)[1]
    X_max = df.iloc[:, 0].max()  # макс значение по Х
    X_min = df.iloc[:, 0].min()  # мин значение по Х

    gamma_min = k * X_min + b + sigma_counter(df, inner_limit)[0]
    gamma_max = k * X_min + b + sigma_counter(df, inner_limit)[1]
    beta_min = k * X_max + b + sigma_counter(df, inner_limit)[0]
    beta_max = k * X_max + b + sigma_counter(df, inner_limit)[1]

    x_in_down, y_in_down = [X_min, X_max], [gamma_min.item(),beta_min.item()]
    x_in_up, y_in_up = [X_min, X_max], [gamma_max.item(),beta_max.item()]

    return sigmaDown, sigmaUp,x_in_down, y_in_down,x_in_up, y_in_up


def linear_function_visualization(x, y, a, b, r2, get_report, x_name, y_name, test_name):
    x_trend = np.linspace(np.min(x), np.max(x), 100)
    y_trend = a * x_trend + b
    sigmaDown, sigmaUp, x_in_down, y_in_down,x_in_up, y_in_up = bourders_initializer(x, y)
    wrong_values1 = []
    wrong_values2 = []

    # Построение кроссплота
    plt.title(test_name)
    plt.scatter(x, y, color='b', label='Данные')
    plt.plot(x_trend, y_trend, color='r', label='Линия тренда')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.legend()

    for i in range(x.size):
        if not(a * x[i] + b + sigmaDown < y[i] < a * x[i] + b + sigmaUp):
            wrong_values2.append(i)
            plt.scatter(x[i], y[i], color='r')

    line1, = plt.plot(x_in_down, y_in_down, marker='o', label='inner_down', color='C2')
    line2, = plt.plot(x_in_up, y_in_up, marker='o', label='inner_up', color='C2')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2), line2: HandlerLine2D(numpoints=2)})
    equation = f'y = {a:.2f}x + {b:.2f}, r2={r2:.2f}'  # Форматирование чисел до двух знаков после запятой
    plt.text(np.min(x), np.mean(y), equation)
    plt.savefig(f"report\\{test_name}")
    if get_report:
        plt.show()
    plt.close()
    return wrong_values1, wrong_values2


def expon_function_visualization(x, y, a, b, r2, get_report, x_name, y_name, test_name):
    data = pd.DataFrame({'x': x, 'y': y})

    data['pred_val'] = a * np.exp(b * data['x'])

    condition = (data['y'] + data['pred_val'] * 0.03) < data['pred_val']

    filtered_data = data[condition]

    wrong_values1 = filtered_data.index.tolist()
    wrong_values2 = filtered_data.index.tolist()

    x_trend = np.linspace(np.min(x), np.max(x), 100)
    y_trend = a * x_trend + b
    plt.title(test_name)
    y_pred = a * np.exp(x * b)

    # Окрашиваем точки, которые не соответствуют линии тренда, в красный

    plt.scatter(x, y, color='b', label='Данные')
    plt.plot(x_trend, y_trend, color='r', label='Линия тренда')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.legend()
    for x, y, pred_val in zip(x, y, y_pred):
        if y + (pred_val * 0.03) < pred_val or x + (pred_val * 0.03) < pred_val:
            plt.scatter(x, y, color='r')
    equation = f'y = {a:.2f}*exp({b:.2f}*x), r2={r2:.2f}'
    plt.text(np.mean(x), np.min(y), equation, ha='center', va='bottom')
    plt.savefig(f"report\\{test_name}")
    if get_report:
        plt.show()
    plt.close()
    return wrong_values1, wrong_values2


def logarithmic_function_visualization(x, y, a, b, r2, get_report, x_name, y_name, test_name):
    data = pd.DataFrame({'x': x, 'y': y})

    data['pred_val'] = a * np.log(data['x']) + b

    condition = (data['y'] + data['pred_val'] * 0.03) < data['pred_val']

    filtered_data = data[condition]

    wrong_values1 = filtered_data.index.tolist()
    wrong_values2 = filtered_data.index.tolist()

    x_trend = np.linspace(np.min(x), np.max(x), 100)
    y_trend = a * x_trend + b

    # Построение кроссплота
    plt.title(test_name)
    y_pred = a * np.log(x) + b

    plt.scatter(x, y, color='b', label='Данные')
    plt.plot(x_trend, y_trend, color='r', label='Линия тренда')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.legend()
    # Окрашиваем точки, которые не соответствуют линии тренда, в красный
    for x, y, pred_val in zip(x, y, y_pred):
        if y + (pred_val * 0.03) < pred_val or x + (pred_val * 0.03) < pred_val:
            plt.scatter(x, y, color='r')
    equation = f'y = {a:.2f}*ln(x)+{b:.2f}, r2={r2:.2f}'
    plt.text(np.mean(x), np.min(y), equation, ha='center', va='bottom')
    plt.savefig(f"report\\{test_name}")
    if get_report:
        plt.show()
    plt.close()
    return wrong_values1, wrong_values2


def remove_nan_pairs(array1, array2):
    new_array1 = []
    new_array2 = []
    index_mapping = {}
    if array1 is not None and array2 is not None:
        try:
            for i in range(array1.size):
                if not (np.isnan(array1[i]) or np.isnan(array2[i])):
                    new_array1.append(array1[i])
                    new_array2.append(array2[i])
                    index_mapping[len(new_array1) - 1] = i
        except:
            ...
    return np.array(new_array1), np.array(new_array2), index_mapping


def remap_wrong_values(wrong_values, index_mapping):
    remapped_values = [index_mapping.get(value, value) for value in wrong_values]
    return remapped_values


def find_test_methods_with_params(params, modul):
    test_methods_with_params = {}

    for method_name, method in inspect.getmembers(modul):
        if inspect.ismethod(method) and method_name.startswith("test"):
            description = inspect.getdoc(method)
            if description:
                # Используем регулярное выражение для извлечения параметров из описания метода
                matches = re.findall(r'Required data:\s*(.*)', description)

                if matches:
                    method_params = [param.strip() for param in matches[0].split(";")]
                    # Проверяем, есть ли переданные параметры в описании метода
                    if set(method_params).issubset(params) and set(method_params).issubset(params):
                        for i in range(len(method_params)):
                            try:
                                test_methods_with_params[method_params[i]].append(method_name)
                            except KeyError:
                                test_methods_with_params[method_params[i]] = []
                                test_methods_with_params[method_params[i]].append(method_name)

    return test_methods_with_params
