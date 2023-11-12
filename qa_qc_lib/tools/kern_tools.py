#########################################################################################################
#######################| РЕАЛИЗАЦИЯ ВСПОМОГАТЕЛЬНЫХ ФУНКЦИЙ QA_QC_kern |##############################
#########################################################################################################
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from sklearn.linear_model import LinearRegression


def sigma_counter(flat_values, sigma_count=1):
    """
        Функция поиска значений сигм для определения доверительного коридора неопределенности

        Args:
        flat_values(np.array): значения, полученные вычитанием аппроксимированных значений линейной регрессии
        sigma_count(int): требуемое количество сигм

        Returns:
        sigmaDown(np.ndarray) -нижняя граница доверительного коридора
        sigmaUp(np.ndarray) -верхняя граница доверительного коридора
    """

    flat_values = flat_values[~np.isnan(flat_values)]
    mean_value = np.mean(flat_values)
    std_value = np.std(flat_values)
    return mean_value - sigma_count * std_value, mean_value + sigma_count * std_value


def linear_regressor(x, y):
    """
    Функция поиска коэффициентов k и b уравнения k*x + b

    Args:
        x(np.array(int)): массив с X значениями
        y(np.array(int)): массив с Y значениями

    Returns:
        k(int): коэффицент наклона
        b(int): коэффицент сдвига от начала координат
    """

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    linear_regressor = LinearRegression()  # создание объекта класса
    linear_regressor.fit(x, y)  # выполнение линейной регрессии

    # Коэффициенты линейной регрессии
    k = linear_regressor.coef_
    b = linear_regressor.intercept_

    return k, b


def bourders_initializer(x, y, inner_limit=1):
    """
    Функция поиска граничных значений неопределенностей.

    Args:
        x(np.array(int)): массив с X значениями
        y(np.array(int)): массив с Y значениями
        inner_limit(int): количество сигм для внутренней границы неопр

    Returns:
        sigmaDown(np.ndarray): нижняя граница доверительного коридора
        sigmaUp(np.ndarray): верхняя граница доверительного коридора
        x_in_down(int): значение нижней границы по X
        y_in_down(int): значение нижней границы по Y
        x_in_up(int): значение нижней границы по X
        y_in_up(int): значение верхней границы по Y
    """

    k, b = linear_regressor(x, y)
    flat_values = y - (k * x + b)
    sigmaDown, sigmaUp = sigma_counter(flat_values, inner_limit)

    X_max = x.max()
    X_min = x.min()

    gamma_min = k * X_min + b + sigmaDown
    gamma_max = k * X_min + b + sigmaUp
    beta_min = k * X_max + b + sigmaDown
    beta_max = k * X_max + b + sigmaUp

    x_in_down, y_in_down = [X_min, X_max], [gamma_min.item(), beta_min.item()]
    x_in_up, y_in_up = [X_min, X_max], [gamma_max.item(), beta_max.item()]

    return sigmaDown, sigmaUp, x_in_down, y_in_down, x_in_up, y_in_up


def dropdown_search(x, y, a, b):
    """
    Функция для вычисления выпадающих точек

    Args:
        x(np.array(int)): массив с X значениями
        y(np.array(int)): массив с Y значениями
        a(int): коэффицент наклона
        b(int): коэффицент сдвига от начала координат

    Returns:
        wrong_values_indices(np.ndarray(int)) - индексы выпадающих значений
    """

    sigmaDown, sigmaUp, x_in_down, y_in_down, x_in_up, y_in_up = bourders_initializer(x, y)

    wrong_values_indices = ~((a * x + b + sigmaDown < y) & (y < a * x + b + sigmaUp))
    # Преобразование  wrong_values_indices в одномерный массив
    wrong_values_indices = np.squeeze(wrong_values_indices)

    # Устанавливаем значения 1 для выпадающих точек
    return wrong_values_indices


def linear_function_visualization(x, y, a, b, r2, get_report, x_name, y_name, test_name,
                                  wrong_values_indices):
    """
    Функция для визуализации линии тренда, отображения выпадающих точек и доверительного коридора

    Args:
        x(np.array(int)): массив с X значениями
        y(np.array(int)): массив с Y значениями
        a(int): коэффицент наклона
        b(int): коэффицент сдвига от начала координат
        r2(int): коэффицент детерминации r2
        get_report(bool): флаг для отображения графика
        x_name(string): название оси X
        y_name(string): название оси Y
        test_name(string): название теста
        wrong_values_indices(np.ndarray):индексы выпадающих точек

    Returns:
        plt(file) - график зависимости
    """

    _, _, x_in_down, y_in_down, x_in_up, y_in_up = bourders_initializer(x, y)

    x_trend = np.linspace(np.min(x), np.max(x), 100)
    y_trend = a * x_trend + b

    # Построение кроссплота
    plt.title(test_name)
    plt.scatter(x, y, color='b', label='Данные')
    plt.plot(x_trend, y_trend, color='r', label='Линия тренда')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.legend()

    # Визуализация неправильных значений
    plt.scatter(x[wrong_values_indices], y[wrong_values_indices], color='r')

    line1, = plt.plot(x_in_down, y_in_down, marker='o', label='inner_down', color='C2')
    line2, = plt.plot(x_in_up, y_in_up, marker='o', label='inner_up', color='C2')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2), line2: HandlerLine2D(numpoints=2)})
    equation = f'y = {a:.2f}x + {b:.2f}, r2={r2:.2f}'  # Форматирование чисел до двух знаков после запятой
    plt.text(np.min(x), np.mean(y), equation)
    plt.savefig(f"data\\{test_name}")
    if get_report:
        plt.show()
    plt.close()
