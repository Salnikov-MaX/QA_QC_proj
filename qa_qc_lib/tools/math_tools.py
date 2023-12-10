#########################################################################################################
##################№###| РЕАЛИЗАЦИЯ ВСПОМОГАТЕЛЬНЫХ МАТЕМАТИЧЕСКИХ ФУНКЦИЙ  |#############################
#########################################################################################################

import numpy as np
from sklearn.linear_model import LinearRegression


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


def sameSign(x: float, y: float):
    """ Функция сравнивает знаки двух чисел

            Args:
                x (float): первое число \n
                y (float): второе число \n
            Returns:
                bool: True - знаки одинаковые \n
                      False - знаки разные
                
        """
    if (x*y > 0):
        return True
    else:
        return False 


def linear_dependence_function(x, y):
    """
    Строит линию тренда по линейной зависимости

    Args:
        x (np.ndarray): Массив чисел.
        y( np.ndarray): Массив чисел.

    Returns:
        a(int): коэффициент наклона
        b(int): коэффициент сдвига от начала координат
    """
    x = list(x)
    y = list(y)
    coefficients = np.polyfit(x, y, 1)
    a = coefficients[0]
    b = coefficients[1]
    return a, b


def exponential_function(x, y):
    """
    Строит линию тренда по экспоненциальной зависимости

    Args:
        x (numpy.ndarray): Массив чисел.
        y( numpy.ndarray): Массив чисел.

    Returns:
        a: коэффициент наклона
        b: коэффициент сдвига от начала координат
    """
    x = list(x)
    y = list(y)
    coefficients = np.polyfit(x, np.log(y), 1)
    a = coefficients[0]
    b = coefficients[1]
    return a, b


def logarithmic_function(x, y):
    """
    Строит линию тренда по логарифмической зависимости

    Args:
        x (numpy.ndarray): Массив чисел.
        y( numpy.ndarray): Массив чисел.

    Returns:
        a: коэффицент наклона
        b: коэффицент сдвига от начала координат
    """
    x_log = np.log(x)
    y_log = y

    # Линейная регрессия в пространстве логарифмов
    model = np.polyfit(x_log, y_log, 1)
    a = model[0]
    b = model[1]
    return a, b


#########################################################################################################
#######################################| ПРОЕКТ "АНОМАЛИИ" |#############################################
#########################################################################################################


class Anomaly_Search_Stat_Methods():
    def __init__(self, data: np.array) -> None:
        """
        Инициализация класса с данными.
        
        Args:
            data (np.array): Временной ряд.
        """
        self.data = data

    def __mody_mean(self, mean_list: np.array, index_list: list) -> float:
        """
        Вычислить модифицированное среднее значение в соответствии с index_list.
        Если индекс равен 0, то элемент не участвует в рассчетах.
        
        Args:
            mean_list (np.array): Список средних значений.
            index_list (list): Список индексов.
        
        Returns:
            float: Модифицированное среднее значение.
        """
        mean_list, index_list = np.array(mean_list), np.array(index_list)
        return np.sum(mean_list * index_list) / np.sum(index_list)

    def __mody_std(self, std_list: np.array, index_list: list) -> float:
        """
        Вычислить модифицированное стандартное отклонение в соответствии с index_list.
        Если индекс равен 0, то элемент не участвует в рассчетах.

        Args:
            std_list (np.array): Список значений стандартного отклонения.
            index_list (list): Список индексов.

        Returns:
        float: Модифицированное стандартное отклонение.
        """
        std_list, index_list = np.array(std_list), np.array(index_list)
        m_std_mean = self.__mody_mean(std_list, index_list)
        m_std = np.sum((std_list - m_std_mean) ** 2 * index_list)
        return np.sqrt(m_std / (np.sum(index_list) - 1))

    def __mody_max(self, max_list: np.array, index_list: list) -> float:
        """
        Найти максимальное значение из max_list, соответствующее 1 в index_list.

        Args:
            max_list (np.array): Список максимальных значений.
            index_list (list): Список индексов.

        Returns:
            float: Максимальное значение, соответствующее 1 в index_list.
        """
        filtered_maxes = [max_val for idx, max_val in enumerate(max_list) if index_list[idx] == 1]
        return max(filtered_maxes)

    def __mody_min(self, min_list: np.array, index_list: list) -> float:
        """
        Найти минимальное значение из min_list, соответствующее 1 в index_list.

        Args:
            min_list (np.array): Список минимальных значений.
            index_list (list): Список индексов.

        Returns:
            float: Минимальное значение, соответствующее 1 в index_list.
        """
        filtered_mins = [min_val for idx, min_val in enumerate(min_list) if index_list[idx] == 1]
        return min(filtered_mins)

    def __sub_function_26(self, data_segment: np.array, start_idx: int, anomaly_counts: list, direction=1,
                          switch_direction=1) -> list:
        """
        Подфункция для определения аномалий в сегменте данных.

        Args:
            data_segment (np.array): Сегмент данных для анализа.
            start_idx (int): Начальный индекс сегмента данных.
            anomaly_counts (list): Список с количеством аномалий для каждой точки данных (весь времянной ряд).
            direction (int, optional): Направление анализа. По умолчанию 1.
            switch_direction (int, optional): Переключатель направления. Принимает значения 1 или -1. По умолчанию 1.

        Returns:
            list: Обновленный список anomaly_counts.
        """
        # Проверка значения switch_direction
        assert switch_direction in [1, -1], "Invalid value for 'switch_direction'. Expected '1' or '-1'."

        # Константы для определения t-альфа в зависимости от размера данных
        SMIRNOV_GRABBS_CONSTANTS = [0, 0, 0, 1.15, 1.46, 1.67, 1.82, 1.94, 2.03, 2.11, 2.18, 2.23, 2.29,
                                    2.33, 2.37, 2.41, 2.44, 2.48, 2.50, 2.53, 2.56, 2.58, 2.6, 2.62, 2.64, 2.66]

        active_elements = [1] * len(data_segment)
        iteration_count, max_iterations = 0, len(data_segment)

        while direction <= 2 and iteration_count < max_iterations:
            # В зависимости от направления выбираем экстремум
            segment_extremum = self.__mody_min(data_segment,
                                               active_elements) if switch_direction == -1 else self.__mody_max(
                data_segment, active_elements)

            elements_count = np.sum(active_elements)
            if elements_count >= 26:
                t_alpha = 0.2849 * np.log(elements_count) + 1.9517
            elif 1.5 <= elements_count < 26:
                t_alpha = SMIRNOV_GRABBS_CONSTANTS[elements_count]
            else:
                return anomaly_counts

            if self.__mody_std(data_segment, active_elements) == 0:
                return anomaly_counts

            t_calculated = abs(segment_extremum - self.__mody_mean(data_segment, active_elements)) / self.__mody_std(
                data_segment, active_elements)

            # Если t-рассчитанное больше t-альфа, обновляем индексы аномалий и сбрасываем направление
            if t_calculated > t_alpha:
                anomaly_positions = np.where(data_segment == segment_extremum)[0]
                for pos in anomaly_positions:
                    anomaly_counts[start_idx + pos] += 1
                    active_elements[pos] = 0
                switch_direction *= -1
                direction = 0
            else:
                # Иначе меняем направление и увеличиваем счетчик направления
                switch_direction *= -1
                direction += 1

        return anomaly_counts

    def find_anomalies_with_window_smirnov(self, shld=13) -> list:
        """
        Поиск аномалий в данных с использованием оконного метода Смирнова.

        Args:
            shld (int, optional): Размер окна. По умолчанию 13.

        Returns:
            list: Список с количеством аномалий для каждой точки данных.
        """
        assert shld > 1.5, "Window size is too small. Minimum window size is 3."

        data = self.data
        window_size, n = shld * 2, len(data)
        anomalies_count = [0] * n  # Список для отслеживания количества аномалий для каждой точки данных.

        for i in range(n + window_size - 1):
            # В зависимости от позиции, размер окна может изменяться
            current_window_size = min(i + 1, window_size, 2 * n - window_size - i)
            start = max(0, i - window_size + 1)  # Начальный индекс текущего окна
            window_data = data[start:start + current_window_size]
            anomalies_count = self.__sub_function_26(window_data, start, anomalies_count)

        return anomalies_count

    def find_anomalies_with_sigma_3(self, shld=13) -> list:
        """
        Определение выбросов во временном ряду с использованием скользящего окна и правила 3-х сигм.

        Args:
            data (numpy.ndarray): временной ряд.
            window_size (int): размер окна.

        Returns:
            anomalies_count (list): маска, отражающая, сколько раз каждая точка была принята за аномалию.
        """
        data = self.data
        n, window_size = len(data), shld * 2
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
                if abs(value - mean_val) > 3 * std_val:
                    anomalies_count[idx] += 1

        return anomalies_count

    def find_anomalies_with_window_iqr(self, shld=13) -> list:
        """
        Обнаруживает аномалии в временном ряду, используя скользящее окно с методом интерквартильного размаха (IQR).

        Args:
            data (numpy.ndarray): Данные временного ряда.
            window_size (int, необязательный): Размер скользящего окна. По умолчанию 13.

        Returns:
            list: Список с количеством аномалий для каждой точки данных
        """
        data = self.data
        n, window_size = len(data), shld * 2
        anomalies_count = [0] * n  # Список для отслеживания количества аномалий для каждой точки данных.

        for i in range(n + window_size - 1):
            # В зависимости от позиции, размер окна может увеличиваться в начале, 
            # быть постоянным в середине и уменьшаться в конце.
            current_window_size = min(i + 1, window_size, 2 * n - window_size - i)
            start = max(0, i - window_size + 1)  # Начальный индекс для текущего окна

            # Извлекаем данные внутри текущего окна
            window_data = data[start:start + current_window_size]

            if len(window_data) > 0:
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

        return anomalies_count

    def detect_anomalies(self, anomalies_count: list, shld=13, method=1, threshold_fraction=0.33) -> list:
        """
        Возвращает список, где 0 означает "нормальное значение" и 1 "аномальное значение".

        Args:
            anomalies_count (list): Входные данные.
            shld (int, optional): Размер плеча (половина окна). По умолчанию 13.
            method (int, optional): Метод для обнаружения аномалий.
                                    1: Обнаруживает аномалии на основе правила трех сигм для всего набора данных.
                                    2: Значения, которые признаны аномальными более чем 2 * shld * threshold_fraction раз.
            threshold_fraction (float, optional): Доля от 2 * shld для метода 2.

        Returns:
            list: Список, где 0 означает "нормальное значение" и 1 "аномальное значение".
        """
        anomalies = [0] * len(anomalies_count)

        if method == 1:
            mean_val, std_dev = np.mean(anomalies_count), np.std(anomalies_count)
            for i in range(shld, len(anomalies_count) - shld):
                if abs(anomalies_count[i] - mean_val) > 3 * std_dev:
                    anomalies[i] = 1
        elif method == 2:
            threshold = 2 * shld * threshold_fraction
            for i, value in enumerate(anomalies_count):
                if value >= threshold:
                    anomalies[i] = 1

        return anomalies

    def find_anomalies(self, shld=13, method=1, threshold_fraction=0.33) -> list:
        """
        Обнаруживает аномалии в данных, комбинируя результаты трех различных методов обнаружения:
        1) Метод трех сигм.
        2) Метод интерквартильного размаха.
        3) Метод Смирнова.
        
        Args:
            shld (int, optional): Размер плеча (половина окна). По умолчанию 13.
            method (int, optional): Метод для обнаружения аномалий для каждого подметода.
                                    1: Обнаруживает аномалии на основе правила трех сигм для всего набора данных.
                                    2: Значения, которые признаны аномальными более чем 2 * shld * threshold_fraction раз.
            threshold_fraction (float, optional): Доля от 2 * shld для метода 2.

        Returns:
            list: Список, где True означает "аномальное значение" и False "нормальное значение".
        """
        # Список методов поиска аномалий. При желании возможно расширить данный список
        anomaly_detectors = [
            self.find_anomalies_with_sigma_3,
            self.find_anomalies_with_window_iqr,
            self.find_anomalies_with_window_smirnov
        ]

        combined_result = np.zeros(len(self.data), dtype=int)

        # Применяем каждый метод детекции аномалий
        for detector in anomaly_detectors:
            anomalies_count = detector(shld=shld)
            detected_anomalies = self.detect_anomalies(anomalies_count=anomalies_count,
                                                       shld=shld, method=method,
                                                       threshold_fraction=threshold_fraction)
            combined_result += detected_anomalies

        # Возвращаем список с обнаружением аномалий (если точка была признана аномальной преобладающим количеством методов)
        return list(combined_result > len(anomaly_detectors) / 2)
