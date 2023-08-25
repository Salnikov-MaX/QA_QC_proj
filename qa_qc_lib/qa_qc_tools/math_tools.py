#########################################################################################################
##################№###| РЕАЛИЗАЦИЯ ВСПОМОГАТЕЛЬНЫХ МАТЕМАТИЧЕСКИХ ФУНКЦИЙ  |#############################
#########################################################################################################

import numpy as np

def compute_variance(arr) -> float:
    """
    Вычисляет дисперсию массива чисел.

    Args:
        arr (numpy.ndarray): Массив чисел.

    Returns:
        float: Дисперсия массива.
    """    
    mean_val = np.mean(arr)
    variance = np.mean((arr - mean_val) ** 2)
    return variance


#########################################################################################################
#######################################| ПРОЕКТ "АНОМАЛИИ" |#############################################
#########################################################################################################


def find_anomalies_with_sigma_3(data, window_size=13):
    """
    Определение выбросов во временном ряду с использованием скользящего окна и правила 3-х сигм.

    Args:
        data (numpy.ndarray): временной ряд.
        window_size (int): размер окна.

    Returns:
        anomalies_count (list): маска, отражающая, сколько раз каждая точка была принята за аномалию.
    """
    n = len(data)
    anomalies_count = [0] * n

    for i in range(2 * n - window_size + 1):
        # В начале окно растет, затем уменьшается, для того чтобы каждая точка побывала в окне равное количество раз
        current_window_size = min(i + 1, window_size, 2 * n - window_size - i)
        start = max(0, i - window_size + 1)

        window_data = data[start:start + current_window_size]
        mean_val = np.mean(window_data)
        std_val = np.std(window_data)

        for j, value in enumerate(window_data):
            idx = start + j
            if abs(value-mean_val) > 3*std_val:
                anomalies_count[idx] += 1

    return anomalies_count


def find_anomalies_with_window_iqr(data, window_size=13):
    
    """
    Обнаруживает аномалии в временном ряду, используя скользящее окно с методом интерквартильного размаха (IQR).

    Args:
        data (numpy.ndarray): Данные временного ряда.
        window_size (int, необязательный): Размер скользящего окна. По умолчанию 13.

    Returns:
        list: Список с количеством аномалий для каждой точки данных.a
    """

    n = len(data)
    anomalies_count = [0] * n  # Список для отслеживания количества аномалий для каждой точки данных.
    is_anomaly = [False] * n  # Список для отслеживания, считается ли точка данных аномалией или нет.

    for i in range(n + window_size - 1):
        # В зависимости от позиции, размер окна может увеличиваться в начале, 
        # быть постоянным в середине и уменьшаться в конце.
        current_window_size = min(i + 1, window_size, 2 * n - window_size - i)
        start = max(0, i - window_size + 1)  # Начальный индекс для текущего окна
        
        # Извлекаем данные внутри текущего окна
        window_data = data[start:start + current_window_size]

        # Вычисляем первый и третий квартили для текущего окна
        Q1 = np.percentile(window_data, 25)
        Q3 = np.percentile(window_data, 75)
        IQR = Q3 - Q1  # Вычисляем интерквартильный размах

        # Определяем верхний и нижний пороги для обнаружения аномалий
        upper_bound = Q3 + 1.5 * IQR
        lower_bound = Q1 - 1.5 * IQR

        # Проходим через данные окна для определения аномалий
        for j, value in enumerate(window_data):
            idx = start + j
            if value > upper_bound or value < lower_bound:
                anomalies_count[idx] += 1  # Увеличиваем счетчик аномалий для точки данных
                is_anomaly[idx] = True  # Помечаем точку данных как аномалию

    return anomalies_count

