#########################################################################################################
#######################| РЕАЛИЗАЦИЯ ВСПОМОГАТЕЛЬНЫХ ФУНКЦИЙ QA_QC_kern |##############################
#########################################################################################################
import inspect
import re

import numpy as np
from matplotlib import pyplot as plt


def linear_function_visualization(x, y, a, b, r2, get_report, x_name, y_name, test_name):
    wrong_values1 = []
    wrong_values2 = []
    y_pred = a * x + b
    for i in range(len(y)):
        if y[i] < a * x[i] + b:
            wrong_values1.append(i)
            wrong_values2.append(i)

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
    for x, y, pred_val in zip(x, y, y_pred):
        if y + (pred_val * 0.1) < pred_val:
            plt.scatter(x, y, color='r')
    plt.text(np.min(x), np.mean(y), equation)
    plt.savefig(f"report\\{test_name}")
    if get_report:
        plt.show()
    return wrong_values1, wrong_values2


def expon_function_visualization(x, y, a, b, r2, get_report, x_name, y_name, test_name):
    wrong_values1 = []
    wrong_values2 = []
    for i in range(len(x)):
        if y[i] > a * np.exp(b * x[i]):
            wrong_values1.append(x[i])
            wrong_values2.append(y[i])

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
        if y + (pred_val * 0.1) < pred_val:
            plt.scatter(x, y, color='r')
    equation = f'y = {a:.2f}*exp({b:.2f}*x), r2={r2:.2f}'
    plt.text(np.mean(x), np.min(y), equation, ha='center', va='bottom')
    plt.savefig(f"report\\{test_name}")
    if get_report:
        plt.show()
    return wrong_values1, wrong_values2


def logarithmic_function_visualization(x, y, a, b, r2, get_report, x_name, y_name, test_name):
    wrong_values1 = []
    wrong_values2 = []
    for i in range(len(x)):
        if y[i] > a * np.log(x[i]) + b:
            wrong_values1.append(x)
            wrong_values2.append(y)

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
        if y + (pred_val * 0.1) < pred_val:
            plt.scatter(x, y, color='r')
    equation = f'y = {a:.2f}*ln(x)+{b:.2f}, r2={r2:.2f}'
    plt.text(np.mean(x), np.min(y), equation, ha='center', va='bottom')
    plt.savefig(f"report\\{test_name}")
    if get_report:
        plt.show()
    return wrong_values1, wrong_values2


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
