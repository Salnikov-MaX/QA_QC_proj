#########################################################################################################
#######################| РЕАЛИЗАЦИЯ ВСПОМОГАТЕЛЬНЫХ ФУНКЦИЙ QA_QC_kern |##############################
#########################################################################################################
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from sklearn.linear_model import LinearRegression


def sigma_counter(df, how_many_sigmas=1):
    """
        Функция поиска значений сигмы для определения доверительного коридора неопределенности

        Args:
        df(dataFrame): дата фрейм
        how_many_sigmas(int): требуемое количество сигм
        get_report(bool): флаг для получения отчета

        Returns:
        sigmaDown(np.ndarray) -нижняя граница доверительного коридора
        sigmaUp(np.ndarray) -верхняя граница доверительного коридора
    """

    flat_values = df['Flat'].dropna().values  # Исключаем NaN значения
    mean_value = np.mean(flat_values)
    std_value = np.std(flat_values)
    return mean_value - how_many_sigmas * std_value, mean_value + how_many_sigmas * std_value


def linear_regressor(df):
    """
        Функция поиска коэффициентов k и b уравнения k*x + b

        Args:
            df(dataFrame): дата фрейм

        Returns:
            k(int): коэффицент наклона
            b(int): коэффицент сдвига от начала координат
    """
    alf_PS = df.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
    Poro = df.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(alf_PS, Poro)  # perform linear regression

    # The coefficients of linear gerression
    k = linear_regressor.coef_
    b = linear_regressor.intercept_

    return k, b


def bourders_initializer(x, y, inner_limit=1):
    """
    Функция поиска граничных значений неопределенностей для проведения оптимизационного процесса.

    Args:
        x(np.array(int)) - массив с X значениями
        y(np.array(int)) - массив с Y значениями
        inner_limit(int) - количество сигм для внутренней границы неопр

    Returns:
        sigmaDown(np.ndarray) -нижняя граница доверительного коридора
        sigmaUp(np.ndarray) -верхняя граница доверительного коридора
        x_in_down(int) - значение нижней границы по X
        y_in_down(int) - значение нижней границы по Y
        x_in_up(int) - значение нижней границы по X
        y_in_up(int) - значение верхней границы по Y
    """

    df = pd.DataFrame({'X': x, 'Y': y})
    k = linear_regressor(df)[0]  # k of k*x + b
    b = linear_regressor(df)[1]  # b of k*x + b
    df["Flat"] = pd.Series(np.nan, index=df.index)
    df["Flat"] = df.apply(lambda row: row['Y'] - (k * row['X'] + b), axis=1)
    sigmas = sigma_counter(df, inner_limit)
    sigmaDown = sigmas[0]
    sigmaUp = sigmas[1]
    X_max = df.iloc[:, 0].max()  # макс значение по Х
    X_min = df.iloc[:, 0].min()  # мин значение по Х

    gamma_min = k * X_min + b + sigma_counter(df, inner_limit)[0]
    gamma_max = k * X_min + b + sigma_counter(df, inner_limit)[1]
    beta_min = k * X_max + b + sigma_counter(df, inner_limit)[0]
    beta_max = k * X_max + b + sigma_counter(df, inner_limit)[1]

    x_in_down, y_in_down = [X_min, X_max], [gamma_min.item(), beta_min.item()]
    x_in_up, y_in_up = [X_min, X_max], [gamma_max.item(), beta_max.item()]

    return sigmaDown, sigmaUp, x_in_down, y_in_down, x_in_up, y_in_up


def linear_function_visualization(x, y, a, b, r2, get_report, x_name, y_name, test_name):
    """
    Функция для визуализации линии тренда, отображения выпадающих точек и доверительного коридора

    Args:
        x(np.array(int)) - массив с X значениями
        y(np.array(int)) - массив с Y значениями
        a(int) - коэффицент наклона
        b(int) - коэффицент сдвига от начала координат
        r2(int) - коэффицент детерминации r2
        get_report(bool) - флаг для отображения графика
        x_name(string) - название оси X
        y_name(string) - название оси Y
        test_name(string) -название теста


    Returns:
        result(np.ndarray) - массив с выпадающими точками относительно линии тренда
        plt(file) - график зависимости
    """
    x_trend = np.linspace(np.min(x), np.max(x), 100)
    y_trend = a * x_trend + b
    sigmaDown, sigmaUp, x_in_down, y_in_down, x_in_up, y_in_up = bourders_initializer(x, y)

    # Построение кроссплота
    plt.title(test_name)
    plt.scatter(x, y, color='b', label='Данные')
    plt.plot(x_trend, y_trend, color='r', label='Линия тренда')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.legend()

    result = np.zeros_like(x)

    wrong_values_indices = ~((a * x + b + sigmaDown < y) & (y < a * x + b + sigmaUp))

    # Убедимся, что wrong_values_indices - одномерный массив
    wrong_values_indices = np.squeeze(wrong_values_indices)

    # Устанавливаем значения в 1 для неправильных значений
    result = np.where(wrong_values_indices, 1, result)

    # Визуализация неправильных значений
    plt.scatter(x[wrong_values_indices], y[wrong_values_indices], color='r')

    line1, = plt.plot(x_in_down, y_in_down, marker='o', label='inner_down', color='C2')
    line2, = plt.plot(x_in_up, y_in_up, marker='o', label='inner_up', color='C2')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2), line2: HandlerLine2D(numpoints=2)})
    equation = f'y = {a:.2f}x + {b:.2f}, r2={r2:.2f}'  # Форматирование чисел до двух знаков после запятой
    plt.text(np.min(x), np.mean(y), equation)
    plt.savefig(f"report\\{test_name}")
    if get_report:
        plt.show()
    plt.close()
    return result
