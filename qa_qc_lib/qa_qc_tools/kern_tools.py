#########################################################################################################
#######################| РЕАЛИЗАЦИЯ ВСПОМОГАТЕЛЬНЫХ ФУНКЦИЙ QA_QC_kern |##############################
#########################################################################################################
import inspect
import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def linear_function_visualization(x, y, a, b, r2, get_report, x_name, y_name, test_name):
    y_pred = a * x + b
    data = pd.DataFrame({'x': x, 'y': y})
    data['pred_val'] = a * data['x'] + b

    wrong_values1 = data.index[(data['x'] + data['pred_val'] * 0.03) < data['pred_val']].tolist()
    wrong_values2 = data.index[(data['y'] + data['pred_val'] * 0.03) < data['pred_val']].tolist()

    x_trend = np.linspace(np.min(x), np.max(x), 100)
    y_trend = a * x_trend + b

    # Построение кроссплота
    plt.title(test_name)
    plt.scatter(x, y, color='b', label='Данные')
    plt.plot(x_trend, y_trend, color='r', label='Линия тренда')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.legend()
    equation = f'y = {a:.2f}x + {b:.2f}, r2={r2:.2f}'  # Форматирование чисел до двух знаков после запятой
    for x_val, y_val, pred_val in zip(x, y, y_pred):
        if y_val + (pred_val * 0.03) < pred_val or x_val + (pred_val * 0.03) < pred_val:
            plt.scatter(x_val, y_val, color='r')
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
