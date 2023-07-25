import os
import shutil
import datetime
from typing import Any
from accessify import private
import numpy as np
from numpy import int32, float64, int64, float32
from scipy.stats import t
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt


class QA_QC_kern:
    def __init__(self, pas=None, note=None,
                 array_for_first_order_tests=None, kno=None, kp_din=None, kp_plast=None, kp_ef=None, density=None,
                 kv=None, kp_pov=None,
                 perpendicular=None, perpendicular_density=None, kp=None, roof=None, takeout=None,
                 parallel_carbonate=None, perpendicular_carbonate=None, perpendicular_porosity=None,
                 kpp=None, intervals=None, sole=None, percentage=None, outreach_in_meters=None, sampling_depth=None,
                 kvo=None, core_sampling=None, kpr=None, parallel_density=None, parallel_porosity=None, parallel=None,
                 rp=None, pmu=None, rn=None, obplnas=None, minple=None,
                 file_report_name="test_report.txt", file_path="D:\\", file_name="не указан", r2=0.7) -> None:
        """_summary_

        Args:
            data (str): _description_
        """
        self.obplnas = obplnas
        self.minple = minple
        self.kno = kno
        self.pas = pas
        self.kp_din = kp_din
        self.kp_plast = kp_plast
        self.kp_pov = kp_pov
        self.kv = kv
        self.density = density
        self.kp_ef = kp_ef
        self.rn = rn
        self.pmu = pmu
        self.rp = rp
        self.parallel = parallel
        self.parallel_porosity = parallel_porosity
        self.parallel_density = parallel_density
        self.kpr = kpr
        self.core_sampling = core_sampling
        self.kvo = kvo
        self.sampling_depth = sampling_depth
        self.outreach_in_meters = outreach_in_meters
        self.percentage = percentage
        self.sole = sole
        self.intervals = intervals
        self.kpp = kpp
        self.perpendicular_porosity = perpendicular_porosity
        self.perpendicular_carbonate = perpendicular_carbonate
        self.parallel_carbonate = parallel_carbonate
        self.takeout = takeout
        self.roof = roof
        self.kp = kp
        self.perpendicular_density = perpendicular_density
        self.data = array_for_first_order_tests
        self.perpendicular = perpendicular
        self.file_report_name = file_report_name
        self.file_path = file_path
        self.table = note
        self.file_name = file_name
        self.r2 = r2
        self.dt_now = datetime.datetime.now()
        self.file = open(self.file_report_name, "w")

    def __del__(self):
        self.file.close()

    @private
    def check_input(self, array, param_name, test_name):
        """Функция для проверки входных данных для тестов первого порядка
        Проверяет, что на вход подается не нулевой массив массив, содержащий только int и float

            Args:
                self.data (array[T]): входной массив для проверки данных

            Returns:
                bool: результат выполнения теста
        """
        if not isinstance(array, np.ndarray):
            self.file.write(
                f"Тест {test_name} не запускался. Причина {param_name}"
                f" не является массивом. Входной файл {self.file_name}. Дата выполнения {self.dt_now}\n")
            return False
        if len(array) == 0:
            self.file.write(
                f"Тест {test_name} не запускался. Причина {param_name}"
                f" пустой. Входной файл {self.file_name}. Дата выполнения {self.dt_now}\n")
            return False
        for element in array:
            if not isinstance(element, (int32, float64, int64, float32)):
                self.file.write(
                    f"Тест {test_name} не запускался. Причина {param_name}"
                    f" содержит не числовые значения. Входной файл {self.file_name}. Дата выполнения {self.dt_now}\n")
                return False
            if np.isnan(element):
                self.file.write(
                    f"Тест {test_name} не запускался. Причина {param_name}"
                    f" содержит данные типа Nan. Входной файл {self.file_name}. Дата выполнения {self.dt_now}\n")
                return False
        return True

    @private
    def linear_dependence_function(self, x, y):
        coefficients = np.polyfit(x, y, 1)
        a = coefficients[0]
        b = coefficients[1]
        return a, b

    @private
    def exponential_function(self, x, y):
        coefficients = np.polyfit(x, np.log(y), 1)
        a = coefficients[0]
        b = coefficients[1]
        return a, b

    """
    Тесты первого порядка 
    """

    def test_water_saturation(self) -> bool:
        """Функция для проверки тестов
        Кво (остаточная водонасыщенность)
        Кв (водонасыщенность)

        Проверяет, что водонасыщенность лежит в интервале от 0 до 1

            Args:
                self.data (array[int/float]): массив с данными для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """

        if self.check_input(self.data, "array for first order tests", "test_water_saturation"):
            dt_now = datetime.datetime.now()
            wrong_values = []
            result = True
            for i in range(len(self.data)):
                if self.data[i] < 0 or self.data[i] > 1:
                    result = False
                    wrong_values.append(i)
            if not result:
                self.file.write(
                    f"Test 'water_saturation': {result}. Данные с индексом {wrong_values}"
                    f" лежат не в интервале от 0 до 1."
                    f" Входной файл {self.file_name}. Дата выполнения {dt_now}\n")
            else:
                self.file.write(
                    f"Test 'water_saturation': {result}. Входной файл {self.file_name}. Дата выполнения {dt_now}\n")
            return result

    def test_porosity(self) -> bool:
        """Функция для проверки тестов
        Открытая пористость в атмосферных условия
        Открытая пористость в пластовых условиях
        Открытая пористость по гелию
        Открытая пористость по керосину
        Эффективная пористость

        Проверяет, что пористость лежит в интервале от 0 до 47.6

            Args:
                self.data (array[int/float]): массив с данными для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """

        if self.check_input(self.data, "array for first order tests", "test porosity"):
            concerted_data = []
            dt_now = datetime.datetime.now()
            wrong_values = []
            result = True
            for val in self.data:
                if val >= 1:
                    concerted_data.append(val / 100)
                else:
                    concerted_data.append(val)
            for i in range(len(concerted_data)):
                if concerted_data[i] < 0 or concerted_data[i] > 0.476:
                    result = False
                    wrong_values.append(i)
            if not result:
                self.file.write(
                    f"Test 'porosity': {result}.Данные с индексом {wrong_values}"
                    f" лежат не в интервале от 0 до 47.6 . Входной файл {self.file_name}. Дата выполнения {dt_now}\n")
            else:
                self.file.write(f"Test 'porosity': {result}. Входной файл {self.file_name}. Дата выполнения {dt_now}\n")
            return result

    def test_permeability(self) -> dict[str, bool | list[int] | str]:
        """Функция для проверки тестов
        Газопроницаемость параллельно напластованию
        Газопроницаемость с поправкой по Кликенбергу
        Эффективная проницаемость
        Проницаемость по воде

        Проверяет, что проницаемость больше от 0

            Args:
                self.data (array[int/float]): массив с данными для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        if self.check_input(self.data, "array for first order tests", "test permeability"):
            wrong_values = []
            result = True
            dt_now = datetime.datetime.now()
            for i in range(len(self.data)):
                if self.data[i] < 0:
                    result = False
                    wrong_values.append(i)
            if not result:
                self.file.write(
                    f"Test 'permeability': {result}.Данные с индексом {wrong_values} больше нуля. Входной файл"
                    f" {self.file_name}. Дата выполнения {dt_now}\n")
            else:
                self.file.write(
                    f"Test 'permeability': {result}. Входной файл {self.file_name}. Дата выполнения {dt_now}\n")
            return {"result": result, "wrong_values": wrong_values, "file_name": self.file_name, "date": dt_now}

    def test_monotony(self) -> bool:
        """Функция для проверки тестов
        Место отбора керна

        Проверяет, что данные монотонно возрастают

            Args:
                self.data (array[int/float]): массив с данными для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        if self.check_input(self.data, "array for first order tests", "test monotony"):
            dt_now = datetime.datetime.now()
            result = True
            wrong_values = []
            for i in range(len(self.data) - 1):
                if self.data[i + 1] - self.data[i] <= 0:
                    result = False
                    wrong_values.append(i)
            if not result:
                self.file.write(
                    f"Test 'monotony': {result}.Данные с индексом {wrong_values} не монотоны. Входной файл {self.file_name}. Дата выполнения {dt_now}\n")
            else:
                self.file.write(f"Test 'monotony': {result}. Входной файл {self.file_name}. Дата выполнения {dt_now}\n")
            return result

    """
        Тесты второго порядка
    """

    def test_quo_kp_dependence(self) -> bool:
        """Функция для проверки теста
        Зависимость Кво-Кп

        Необходимо построить линию тренда и проверить, что зависимость линейная по функции y=a*x+b, при этом a<0

            Args:
                self.kvo (array[int/float]): массив с данными кво для проверки
                self.kp (array[int/float]): массив с данными кп для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """

        if self.check_input(self.kvo, "kvo", "test quo kp dependence") and \
                self.check_input(self.kp, "kp", "test quo kp dependence"):
            result = True
            a, b = self.linear_dependence_function(self.kvo, self.kp)
            if a >= 0:
                result = False

            self.file.write(
                f"Test 'dependence quo kp': {result}. Входной файл {self.file_name}. Дата выполнения {self.dt_now}\n")
            x_trend = np.linspace(np.min(self.kvo), np.max(self.kvo), 100)
            y_trend = a * x_trend + b

            # Построение кроссплота
            plt.title("test quo kp dependence")
            plt.scatter(self.kvo, self.kp, color='b', label='Данные')
            plt.plot(x_trend, y_trend, color='r', label='Линия тренда')
            plt.xlabel('kvo')
            plt.ylabel('kp')
            plt.legend()
            equation = f'y = {a:.2f}x + {b:.2f}'  # Форматирование чисел до двух знаков после запятой
            plt.text(np.mean(self.kvo), np.min(self.kp), equation, ha='center', va='bottom')
            plt.show()

            return result

    def test_kp_density_dependence(self) -> bool:
        """Функция для проверки тестов
        Зависимость Кп-плотность

        Необходимо построить линию тренда и проверить, что зависимость линейная по функции y=a*x+b, при этом a<0

            Args:
                self.kp (array[int/float]): массив с данными кп для проверки
                self.density (array[int/float]): массив с данными плотностью для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        if self.check_input(self.kp, "kp", "test kp density dependence") and \
                self.check_input(self.density, "density", "test kp density dependence"):

            result = True
            a, b = self.linear_dependence_function(self.kp, self.density)
            if a >= 0:
                result = False

            self.file.write(
                f"Test 'dependence quo kp': {result}. Входной файл {self.file_name}. Дата выполнения {self.dt_now}\n")
            x_trend = np.linspace(np.min(self.kp), np.max(self.density), 100)
            y_trend = a * x_trend + b

            # Построение кроссплота
            plt.title("test kp density dependence")
            plt.scatter(self.kp, self.density, color='b', label='Данные')
            plt.plot(x_trend, y_trend, color='r', label='Линия тренда')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            equation = f'y = {a:.2f}x + {b:.2f}'
            plt.text(np.mean(self.kp), np.min(self.density), equation, ha='center', va='bottom')
            plt.show()

            return result

    def test_kvo_kp_din_dependence(self) -> bool:
        """Функция для проверки тестов
        Зависимость Кво-Кпдин

        Необходимо построить линию тренда и проверить, что зависимость линейная по функции y=a*x+b, при этом a<0

            Args:
                self.kvo (array[int/float]): массив с данными кво для проверки
                self.kp_din (array[int/float]): массив с данными кпдин для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        if self.check_input(self.kvo, "kvo", "test kvo kp din dependence") and \
                self.check_input(self.kp_din, "kp din", "test kvo kp din dependence"):

            result = True
            a, b = self.linear_dependence_function(self.kvo, self.kp_din)
            if a >= 0:
                result = False

            self.file.write(
                f"Test 'dependence quo kp': {result}. Входной файл {self.file_name}. Дата выполнения {self.dt_now}\n")
            x_trend = np.linspace(np.min(self.kvo), np.max(self.kp_din), 100)
            y_trend = a * x_trend + b

            # Построение кроссплота
            plt.title("test kvo kp din dependence")
            plt.scatter(self.kvo, self.kp_din, color='b', label='Данные')
            plt.plot(x_trend, y_trend, color='r', label='Линия тренда')
            plt.xlabel('kvo')
            plt.ylabel('kp din')
            plt.legend()
            equation = f'y = {a:.2f}x + {b:.2f}'
            plt.text(np.mean(self.kvo), np.min(self.kp_din), equation, ha='center', va='bottom')
            plt.show()

            return result

    def test_obblnas_kp_dependence(self) -> bool:
        """Функция для проверки теста
        Обплнас-Кп

        Необходимо построить линию тренда и проверить,зависимость по функции y=a1*x+b1, при этом a1<a2,
         где а2 - коэффициент из зависимости Минпл-Кп


            Args:
                self.kp (array[int/float]): массив с данными Обплнас для проверки
                self.obblnas (array[int/float]): массив с данными кп для проверки
                self.minple (array[int/float]): массив с данными Минпл для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        if self.check_input(self.minple, "minple", "test obblnas kp dependence") and \
                self.check_input(self.kp, "kp", "test obblnas kp dependence") and \
                self.check_input(self.obplnas, "obplnas", "test obblnas kp dependence"):
            coeffs1 = np.polyfit(self.minple, self.kp, 1)
            a1, b1 = coeffs1[0], coeffs1[1]
            trend_line1 = np.polyval(coeffs1, self.minple)

            coeffs2 = np.polyfit(self.obplnas, self.kp, 1)
            a2, b2 = coeffs2[0], coeffs2[1]
            trend_line2 = np.polyval(coeffs2, self.obplnas)

            result = True
            if a1 >= a2:
                result = False

            self.file.write(
                f"Test 'dependence obblnas kp': {result}. Входной файл {self.file_name}."
                f" Дата выполнения {self.dt_now}\n")
            plt.title("test obblnas kp dependence")
            plt.scatter(self.obplnas, self.kp, color='red', label='Обплнас-Кп')
            plt.scatter(self.minple, self.kp, color='blue', label='Минпл-Кп')
            plt.plot(self.obplnas, trend_line1, color='red', label=f'Обплнас-Кп: y={a1:.2f}x + {b1:.2f}')
            plt.plot(self.minple, trend_line2, color='blue', label=f'Минпл-Кп: y={a2:.2f}x + {b2:.2f}')
            plt.xlabel('obplnas')
            plt.ylabel('kp')
            plt.legend()
            plt.grid(True)
            equation = f'y = {a1:.2f}x + {b1:.2f}'
            plt.text(np.mean(self.minple), np.min(self.kp) + 0.5, equation, ha='center', va='bottom')
            equation = f'y = {a2:.2f}x + {b2:.2f}'
            plt.text(np.mean(self.obplnas), np.min(self.kp), equation, ha='center', va='bottom')
            plt.show()

            return result

    def test_minple_kp_dependence(self) -> bool:
        """Функция для проверки теста
        Минпл-Кп

        Необходимо построить линию тренда и проверить,зависимость по функции y=a1*x+b1, при этом a1<a2,
        где а1 - коэффициент из зависимости Обплнас-Кп

            Args:
                self.kp (array[int/float]): массив с данными Обплнас для проверки
                self.obblnas (array[int/float]): массив с данными кп для проверки
                self.minple (array[int/float]): массив с данными Минпл для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        if self.check_input(self.minple, "minple array", "test minple kp dependence") and \
                self.check_input(self.kp, "kp", "test minple kp dependence") and \
                self.check_input(self.obplnas, "obplnas_array", "test minple kp dependence"):
            coeffs1 = np.polyfit(self.obplnas, self.kp, 1)
            a1, b1 = coeffs1[0], coeffs1[1]
            trend_line1 = np.polyval(coeffs1, self.obplnas)

            coeffs2 = np.polyfit(self.minple, self.kp, 1)
            a2, b2 = coeffs2[0], coeffs2[1]
            trend_line2 = np.polyval(coeffs2, self.minple)

            result = True
            if a1 <= a2:
                result = False

            self.file.write(
                f"Test 'dependence minple kp': {result}."
                f" Входной файл {self.file_name}. Дата выполнения {self.dt_now}\n")
            plt.title("test minple kp dependence")
            plt.scatter(self.obplnas, self.kp, color='red', label='Обплнас-Кп')
            plt.scatter(self.minple, self.kp, color='blue', label='Минпл-Кп')
            plt.plot(self.obplnas, trend_line1, color='red', label=f'Обплнас-Кп: y={a1:.2f}x + {b1:.2f}')
            plt.plot(self.minple, trend_line2, color='blue', label=f'Минпл-Кп: y={a2:.2f}x + {b2:.2f}')
            plt.xlabel('minple')
            plt.ylabel('kp')
            plt.legend()
            plt.grid(True)
            equation = f'y = {a1:.2f}x + {b1:.2f}'
            plt.text(np.mean(self.minple), np.min(self.kp) + 0.5, equation, ha='center', va='bottom')
            equation = f'y = {a2:.2f}x + {b2:.2f}'
            plt.text(np.mean(self.obplnas), np.min(self.kp), equation, ha='center', va='bottom')
            plt.show()

            return result

    def test_kpf_kpdin_dependence(self) -> bool:
        """Функция для проверки теста
        Зависимость Кпэф-Кпдин

        Необходимо построить линию тренда и проверить,что зависимость линейная по функции y=a*x+b, при этом a>0, b>0

            Args:
                self.x (array[int/float]): массив с данными кпэф для проверки
                self.y (array[int/float]): массив с данными кпдин для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """

        if self.check_input(self.kp_ef, "kp ef", "test kpf kp din dependence") \
                and self.check_input(self.kp_din, "kp din", "test kpf kp din dependence"):

            result = True
            a, b = self.linear_dependence_function(self.kp_ef, self.kp_din)
            if a <= 0 or b <= 0:
                result = False

            self.file.write(
                f"Test 'dependence kpf kpdin': {result}."
                f" Входной файл {self.file_name}. Дата выполнения {self.dt_now}\n")
            x_trend = np.linspace(np.min(self.kp_ef), np.max(self.kp_din), 100)
            y_trend = a * x_trend + b
            plt.title("test kpf kpdin dependence")
            plt.scatter(self.kp_ef, self.kp_din, color='b', label='Данные')
            plt.plot(x_trend, y_trend, color='r', label='Линия тренда')
            plt.xlabel('kp ef')
            plt.ylabel('kp din')
            plt.legend()
            equation = f'y = {a:.2f}x + {b:.2f}'
            plt.text(np.mean(self.kp_ef), np.min(self.kp_din), equation, ha='center', va='bottom')
            plt.show()
            return result

    def test_kp_ef_kp_dependence(self) -> bool:
        """Функция для проверки тестов
        Зависимость Кпэф-Кп

        Необходимо построить линию тренда и проверить,что зависимость линейная по функции y=a*x+b, при этом a>0, b<0

            Args:
                self.kp_ef (array[int/float]): массив с данными кпэф для проверки
                self.kp (array[int/float]): массив с данными кп для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        if self.check_input(self.kp_ef, "kp ef", "test kp ef kp dependence") \
                and self.check_input(self.kp, "kp", "test kp ef kp dependence"):
            result = True
            a, b = self.linear_dependence_function(self.kp_ef, self.kp)
            if a <= 0 or b >= 0:
                result = False

            self.file.write(
                f"Test 'dependence kp ef kp': {result}. Входной файл {self.file_name}. Дата выполнения {self.dt_now}\n")
            x_trend = np.linspace(np.min(self.kp_ef), np.max(self.kp), 100)
            y_trend = a * x_trend + b
            plt.title("test kp ef kp dependence")
            plt.scatter(self.kp_ef, self.kp, color='b', label='Данные')
            plt.plot(x_trend, y_trend, color='r', label='Линия тренда')
            plt.xlabel('kp ef')
            plt.ylabel('kp')
            plt.legend()
            equation = f'y = {a:.2f}x + {b:.2f}'
            plt.text(np.mean(self.kp_ef), np.min(self.kp), equation, ha='center', va='bottom')
            plt.show()
            return result

    def test_kp_kp_din_dependence(self) -> bool:
        """Функция для проверки тестов
        Зависимость Кп-Кпдин

        Необходимо построить линию тренда и проверить,что зависимость линейная по функции y=a*x+b, при этом a>0, b<0

            Args:
                self.kp (array[int/float]): массив с данными кп для проверки
                self.kp_din (array[int/float]): массив с данными кп дин для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        if self.check_input(self.kp, "kp", "test kp kp din dependence") \
                and self.check_input(self.kp_din, "kp din", "test kp kp din dependence"):
            a, b = self.linear_dependence_function(self.kp, self.kp_din)
            result = True
            if a <= 0 or b >= 0:
                result = False

            self.file.write(
                f"Test 'dependence kp ef kp': {result}. Входной файл {self.file_name}. Дата выполнения {self.dt_now}\n")
            x_trend = np.linspace(np.min(self.kp), np.max(self.kp_din), 100)
            y_trend = a * x_trend + b
            plt.title("test kp ef kp dependence")
            plt.scatter(self.kp, self.kp_din, color='b', label='Данные')
            plt.plot(x_trend, y_trend, color='r', label='Линия тренда')
            plt.xlabel('kp')
            plt.ylabel('kp din')
            plt.legend()
            equation = f'y = {a:.2f}x + {b:.2f}'
            plt.text(np.mean(self.kp), np.min(self.kp_din), equation, ha='center', va='bottom')
            plt.show()
            return result

    def test_dependence_kpr_kp(self) -> bool:
        """Функция для проверки тестов
        Зависимость Кпр-Кп

        Необходимо построить линию тренда и проверить,что  зависимость по функции y=a*exp(b*x) при этом b>0

            Args:
                self.kpr (array[int/float]): массив с данными кпр для проверки
                self.kp (array[int/float]): массив с данными кп для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """

        if self.check_input(self.kpr, "kpr", "test dependence kpr kp") and \
                self.check_input(self.kp, "kp", "test dependence kpr kp"):

            result = True
            a, b = self.exponential_function(self.kpr, self.kp)
            if b <= 0:
                result = False

            self.file.write(
                f"Test 'dependence kpr kp': {result}. Входной файл {self.file_name}. Дата выполнения {self.dt_now}\n")
            x_trend = np.linspace(np.min(self.kpr), np.max(self.kp), 100)
            y_trend = a * x_trend + b
            plt.title("test dependence kpr kp")
            plt.scatter(self.kpr, self.kp, color='b', label='Данные')
            plt.plot(x_trend, y_trend, color='r', label='Линия тренда')
            plt.xlabel('kpr')
            plt.ylabel('kp')
            plt.legend()
            equation = f'y = {a:.2f}*exp({b:.2f}*x)'
            plt.text(np.mean(self.kp), np.min(self.kp_din), equation, ha='center', va='bottom')
            plt.show()
            return result

    def test_dependence_kpr_kp_din(self) -> bool:
        """Функция для проверки теста
        Зависимость Кпр-Кпдин

        Необходимо построить линию тренда и проверить,что  зависимость по функции y=a*exp(b*x) при этом b>0

            Args:
                self.kpr (array[int/float]): массив с данными кпр для проверки
                self.kp_din (array[int/float]): массив с данными кпдин для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """

        if self.check_input(self.kpr, "first array for dependency testing", "test dependence kpr kp din") and \
                self.check_input(self.kp_din, "second array for dependency testing", "test dependence kpr kp din"):

            result = True
            a, b = self.exponential_function(self.kpr, self.kp_din)
            if b <= 0:
                result = False

            self.file.write(
                f"Test 'dependence kpc kp': {result}. Входной файл {self.file_name}. Дата выполнения {self.dt_now}\n")
            x_trend = np.linspace(np.min(self.kpr), np.max(self.kp_din), 100)
            y_trend = a * x_trend + b
            plt.title("test dependence kpc kp")
            plt.scatter(self.kpr, self.kp_din, color='b', label='Данные')
            plt.plot(x_trend, y_trend, color='r', label='Линия тренда')
            plt.xlabel('kpr')
            plt.ylabel('kp din')
            plt.legend()
            equation = f'y = {a:.2f}*exp({b:.2f}*x)'
            plt.text(np.mean(self.kp), np.min(self.kp_din), equation, ha='center', va='bottom')
            plt.show()
            return result

    def test_dependence_kvo_kpr(self) -> bool:
        """Функция для проверки тестов
        Зависимость Кво-Кпр

        Необходимо построить линию тренда и проверить,что зависимость по функции y=a*ln(x)+b при этом a>0

            Args:
                self.kvo (array[int/float]): массив с данными кво для проверки
                self.kpr (array[int/float]): массив с данными кпр для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """

        if self.check_input(self.kvo, "kvo", "test dependence kvo kpr") and \
                self.check_input(self.kpr, "kpr", "test dependence kvo kpr"):
            coefficients = np.polyfit(self.kvo, np.exp(self.kpr), 1)
            a, b = coefficients[0], coefficients[1]
            result = True
            if a <= 0:
                result = False

            self.file.write(
                f"Test 'dependence kvo kpr': {result}. Входной файл {self.file_name}. Дата выполнения {self.dt_now}\n")
            x_trend = np.linspace(np.min(self.kvo), np.max(self.kpr), 100)
            y_trend = a * x_trend + b

            # Построение кроссплота
            plt.title("test dependence kvo kpr")
            plt.scatter(self.kvo, self.kpr, color='b', label='Данные')
            plt.plot(x_trend, y_trend, color='r', label='Линия тренда')
            plt.xlabel('kvo')
            plt.ylabel('kpr')
            plt.legend()
            equation = f'y = {a:.2f}*ln(x)+{b:.2f}'
            plt.text(np.mean(self.kvo), np.min(self.kpr), equation, ha='center', va='bottom')
            plt.show()
            return result

    def test_rn_kv_dependencies(self) -> bool:
        """Функция для проверки тестов
        Зависимости Рн-Кв

        Необходимо построить линию тренда и проверить,что зависимость по функции y=b/(kв^n) при этом 1,1<n<5

            Args:
                self.rn (array[int/float]): массив с данными рн для проверки
                self.kv (array[int/float]): массив с данными кв для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        if self.check_input(self.rn, "rn", "test rn kv dependencies") \
                and self.check_input(self.kv, "kv", "test rn kv dependencies"):
            coefficients = np.polyfit(np.log(self.rn), np.log(self.kv), 1)
            b, n = np.exp(coefficients[1]), coefficients[0]
            result = True
            if 1.1 >= n or n >= 5:
                result = False

            self.file.write(
                f"Test 'dependence rn kv': {result}. Входной файл {self.file_name}. Дата выполнения {self.dt_now}\n")
            plt.title("test rn kv dependencies")
            plt.scatter(self.rn, self.kv, color='blue', label='Исходные данные')
            plt.plot(self.rn, b / (self.kv ** n), color='red', label='Линия тренда')
            plt.xlabel('rn')
            plt.ylabel('kv')
            plt.legend()
            equation = f'y = {b:.2f}/(x^{b:.2f})'
            plt.text(np.mean(self.kp), np.min(self.kp_din), equation, ha='center', va='bottom')
            plt.show()
            return result

    def test_rp_kp_dependencies(self) -> bool:
        """Функция для проверки тестов
        Зависимость Рп-Кп

        В данном случае зависимость по функции y=a/(kп^m) при этом m>0. a>0 и a<2,5, 1,1<m<3,8

            Args:
                self.rp (array[int/float]): массив с данными рп для проверки
                self.kp (array[int/float]): массив с данными кп для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        if self.check_input(self.rp, "rp", "test rp kp dependencies") \
                and self.check_input(self.kp, "kp", "test rp kp dependencies"):
            coefficients = np.polyfit(np.log(self.rn), -np.log(self.kp), 1)
            a, m = np.exp(-coefficients[1]), coefficients[0]
            result = True
            if 1.1 >= m or m >= 3.8 or 0 >= a or a >= 2.5:
                result = False

            self.file.write(f"Test 'dependence rp kp ': {result}."
                            f" Входной файл {self.file_name}. Дата выполнения {self.dt_now}\n")

            plt.scatter(self.rp, self.kp, color='blue', label='Исходные данные')
            plt.plot(self.rp, a / (self.kp ** m), color='red', label='Линия тренда')
            plt.title("test rp kp dependencies")
            plt.xlabel('рп')
            plt.ylabel('кп')
            plt.legend()
            equation = f'y = {a:.2f}/(x^{m:.2f})'
            plt.text(np.mean(self.kp), np.min(self.kp_din), equation, ha='center', va='bottom')
            plt.show()

            return result

    def test_general_dependency_checking(self, x, y) -> bool:
        """Функция для проверки теста
        Для всех зависимостей

        Необходимо отбросить очевидные выбросные точки про помощи доверительного интервала
        и распределения Стьюдента. После этого аппроксимировать полученное облако точек и
        оценить R2. Если значение ≥ 0.7, то все ок, тест пройден, если значение меньше,
        то ищем дистанции от линии тренда до каждой из точек, сортируем дистанции по
        убыванию, и запускаем цикл, в которой в каждой итерации которого удаляется самая
        дальняя точка, и считается R2. Цикл проводится до тех пор, пока R2 не станет
        больше 0,7. Если удалили меньше 10% точек, то тест пройден, если больше -
        тест не пройден

            Args:
                x (array[int/float]): массив с данными для проверки
                y (array[int/float]): массив с данными для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        if self.check_input(x, "first array for dependency testing", "test general dependency checking") \
                and self.check_input(y, "second array for dependency testing", "test general dependency checking"):
            alpha = 0.05  # Уровень значимости
            n = len(x)
            dof = n - 2  # Число степеней свободы для распределения Стьюдента

            residuals = y - np.polyval(np.polyfit(x, y, 1), x)
            std_error = np.sqrt(np.sum(residuals ** 2) / dof)

            t_critical = t.ppf(1 - alpha / 2, dof)
            upper_limit = np.polyval(np.polyfit(x, y, 1), x) + t_critical * std_error
            lower_limit = np.polyval(np.polyfit(x, y, 1), x) - t_critical * std_error

            x_filtered = []
            y_filtered = []

            for i in range(n):
                if lower_limit[i] <= y[i] <= upper_limit[i]:
                    x_filtered.append(x[i])
                    y_filtered.append(y[i])

            # Аппроксимация линии тренда на отфильтрованных данных
            coeffs = np.polyfit(x_filtered, y_filtered, 1)
            trend_line = np.polyval(coeffs, x_filtered)

            # Вычисление R2 score
            r2 = r2_score(y_filtered, trend_line)

            # Проверка условия R2 score и удаление точек при несоответствии
            while r2 < self.r2 and len(x_filtered) > 0.9 * n:
                # Вычисление расстояний от линии тренда до каждой точки
                distances = np.abs(y_filtered - np.polyval(coeffs, x_filtered))

                # Сортировка расстояний по убыванию
                sorted_indices = np.argsort(distances)[::-1]

                # Удаление самой дальней точки
                x_filtered.pop(sorted_indices[0])
                y_filtered.pop(sorted_indices[0])

                # Повторное вычисление линии тренда и R2 score
                coeffs = np.polyfit(x_filtered, y_filtered, 1)
                trend_line = np.polyval(coeffs, x_filtered)
                r2 = r2_score(y_filtered, trend_line)

            # Проверка пройденного теста
            if r2 >= 0.7:
                result = True
                self.file.write(
                    f"Test 'general dependency checking': {result}."
                    f" Входной файл {self.file_name}. Дата выполнения {self.dt_now}\n")
                return result
            else:
                result = False
                self.file.write(
                    f"Test 'general dependency checking': {result}."
                    f" Входной файл {self.file_name}. Дата выполнения {self.dt_now}\n")
                return result

    def test_coring_depths_first(self):
        """Функция для проверки теста
        Глубины отбора керна_1

        Необходимо проверить, что подошва вышележащего долбления должна быть выше или равна кровле нижележащего

            Args:
                self.roof (array[int/float]): массив с данными кровли для проверки
                self.sole (array[int/float]): массив с данными подошвы для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        if self.check_input(self.roof, "roof",
                            "test coring depths first") and self.check_input(self.sole,
                                                                             "sole",
                                                                             "test coring depths first"):
            result = True
            for i in range(len(self.roof)):
                if self.sole[i] < self.roof[i]:
                    result = False
            self.file.write(
                f"Test 'coring depths first': {result}. Входной файл {self.file_name}. Дата выполнения {self.dt_now}\n")
            return result

    def test_coring_depths_second(self):
        """Функция для проверки теста
        Глубины отбора керна_2

        Необходимо проверить, что подошва-кровля ≥ выносу в метрах

            Args:
                self.roof (array[int/float]): массив с данными кровли для проверки
                self.sole (array[int/float]): массив с данными подошвы для проверки
                self.takeout (array[int/float]): массив с данными выноса в метрах для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        if self.check_input(self.roof, "roof", "test coring depths second") \
                and self.check_input(self.sole, "sole", "test coring depths second") \
                and self.check_input(self.takeout, "takeout", "test coring depths second"):
            dt_now = datetime.datetime.now()
            result = True
            for i in range(len(self.roof)):
                if self.sole[i] - self.roof[i] > self.takeout[i]:
                    result = False
            self.file.write(
                f"Test 'coring depths second': {result}. Входной файл {self.file_name}. Дата выполнения {dt_now}\n")
            return result

    def test_coring_depths_third(self):
        """Функция для проверки теста
        Глубины отбора керна_3

        Необходимо проверить, что значения выноса в метрах и % должны соответствовать друг другу.

            Args:
                self.intervals (array[[int/float]]): массив с массивамими,
                                                    содержашими начало интервала и конец интервала
                self.percentage (array[int/float]): массив со значениями выноса в процентах

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        if self.check_input(self.intervals, "intervals", "test coring depths third") \
                and self.check_input(self.outreach_in_meters, "outreach_in_meters", "test coring depths third") \
                and self.check_input(self.percentage, "percentage", "test coring depths third"):
            result = True
            for i in range(len(self.intervals)):
                interval_length = max(self.intervals[i]) - min(self.intervals[i])
                displacement_percent_calculated = (self.outreach_in_meters[i] / interval_length) * 100
                if abs(self.percentage[i] - displacement_percent_calculated) > 0.001:
                    result = False

            self.file.write(
                f"Test 'coring depths third': {result}. Входной файл {self.file_name}. Дата выполнения {self.dt_now}\n")
            return result

    def test_coring_depths_four(self):
        """Функция для проверки теста
        Глубины отбора керна_4

        Необходимо проверить, что глубина отбора образца НЕ НИЖЕ фактического выноса керна.

            Args:
                self.sampling_depth (array[int/float]): массив с фактическими выносами керна
                self.core_sampling (array[int/float]): массив с глубинами отбора образцов

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        if self.check_input(self.sampling_depth, "sampling_depth", "test coring depths four") and self.check_input(
                self.core_sampling, "core_sampling", "test coring depths four"):
            result = True
            for i in range(len(self.sampling_depth)):
                if self.core_sampling[i] > self.sampling_depth[i]:
                    result = False

            self.file.write(
                f"Test 'coring depths four': {result}. Входной файл {self.file_name}. Дата выполнения {self.dt_now}\n")
            return result

    def test_data_tampering(self):
        """Функция для проверки теста
        Подлог данных

        Поиск одинаковых (очень близких по значениям) образцов на основе взаимозависимых параметров:
        Кпр=f(Кп); Рп=f(Кп); Рн=f(Кв); Кво=f(Кпр); Кво=f(Кп); плотность (любая)=f(Кп).

            Args:
                self.kpr (array[int/float]): массив с данными кпр для сравнения
                self.kp (array[int/float]): массив с данными кп для сравнения
                self.kvo (array[int/float]): массив с данными кво для сравнения
                self.rp (array[int/float]): массив с данными рп для сравнения
                self.density (array[int/float]): массив с данными всех плотностей для сравнения
                self.rn (array[int/float]): массив с данными рн для сравнения
                self.kv (array[int/float]): массив с данными кв для сравнения

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
                """
        if self.check_input(self.sampling_depth, "sampling_depth", "test coring depths four"):
            result = True
            for i in range(len(self.kpr)):
                for j in range(len(self.kp)):
                    if abs(self.kpr[i] - self.kp[j]) > 0.01:
                        result = False
                for k in range(len(self.kvo)):
                    if abs(self.kpr[i] - self.kvo[k]) > 0.01:
                        result = False
            for i in range(len(self.kp)):
                for j in range(len(self.rp)):
                    if abs(self.kp[i] - self.rp[j]) > 0.01:
                        result = False
                for k in range(len(self.kvo)):
                    if abs(self.kp[i] - self.kvo[k]) > 0.01:
                        result = False
                for d in range(len(self.density)):
                    if abs(self.kp[i] - self.density[d]) > 0.01:
                        result = False
            for i in range(len(self.rn)):
                for k in range(len(self.kv)):
                    if abs(self.rn[i] - self.kv[k]) > 0.01:
                        result = False

            self.file.write(
                f"Test 'data tampering': {result}. Входной файл {self.file_name}. Дата выполнения {self.dt_now}\n")
            return result

    def test_estimated_values_in_core_data(self, pv=1):
        """Функция для проверки теста
        Расчетные величины в данных керна

        необходимо перепроверять фактическим расчетом.
        Кп_эф=Кп*(1-Кво)
        Кп.дин.= Кп*(1-Кво-Кно)
        ρму=ρас+(Кп*ρв)
        Кп_эф, Кп.д., Кп, Кво, Кно берется в д.ед. ρв зачастую принимается равным 1г/см3

            Args:
                self.kp_ef (array[int/float]): массив с данными кпэф для проверки
                self.kp (array[int/float]): массив с данными кп для рассчета
                self.kvo (array[int/float]): массив с данными кво для рассчета
                self.kp_din (array[int/float]): массив с данными кп. дин. для проверки
                self.kno (array[int/float]): массив с данными кно для рассчетов
                self.pmu (array[int/float]): массив с данными рму для проверки
                self.pas (array[int/float]): массив с данными кп для рассчетов
                pv  (const): константа для рассчетов
            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        if self.check_input(self.kp_ef, "kp_ef", "test estimated values in core data") and \
                self.check_input(self.kp, "kp", "test estimated values in core data") and \
                self.check_input(self.kvo, "kvo", "test estimated values in core data") and \
                self.check_input(self.kp_din, "kp_din", "test estimated values in core data") and \
                self.check_input(self.kno, "kno", "test estimated values in core data") and \
                self.check_input(self.pmu, "pmu", "test estimated values in core data") and \
                self.check_input(self.pas, "pas", "test estimated values in core data"):
            result = True
            for i in range(len(self.kp_ef)):
                if self.kp[i] * (1 - self.kvo[i]) != self.kp_ef[i]:
                    result = False

            for i in range(len(self.kp_din)):
                if self.kp[i] * (1 - self.kvo[i] - self.kno[i]) != self.kp_din[i]:
                    result = False

            for i in range(len(self.pmu)):
                if self.pas[i] + (self.kp[i] * pv) != self.pmu[i]:
                    result = False

            self.file.write(
                f"Test 'estimated values in core data': {result}."
                f" Входной файл {self.file_name}. Дата выполнения {self.dt_now}\n")
            return result

    def test_kp_in_surface_and_reservoir_conditions(self):
        """Функция для проверки теста
        Кп в поверхностных и пластовых условиях

        Кп в атмосферных условиях всегда больше чем Кп в пластовых условиях

            Args:
                self.kp_pov (array[int/float]): массив с данными кп в поверхностных условиях для проверки
                self.kp_plast (array[int/float]): массив с данными кп в пластовых условиях для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
                """
        if self.check_input(self.kp_pov, "kp_pov", "test kp in surface and reservoir conditions") and \
                self.check_input(self.kp_plast, "kp_plast", "test kp in surface and reservoir conditions"):
            result = True
            for i in range(len(self.kp_pov)):
                if abs(self.kp_pov[i] - self.kp_plast[i]) > 2:
                    result = False

            self.file.write(
                f"Test 'kp in surface and reservoir conditions': {result}."
                f" Входной файл {self.file_name}. Дата выполнения {self.dt_now}\n")
            return result

    def test_table_notes(self):
        """Функция для проверки теста
        Примечания в таблицах

        есть столбец “Примечания” эту информацию необходимо учитывать

            Args:
                self.table (array[int/float]): массив с данными из таблицы с
                примечанием, при наличии ошибки в массиве будет находиться 1,
                при отсутсвии 0

            Returns:
                array[int]: индексы на которых находятся дефекты
                file: запись результата теста для сохранения состояния
        """
        if self.check_input(self.table, "note", "test table notes"):
            indexes = [i for i, x in enumerate(self.table) if x == 1]
            self.file.write("Test 'table notes': {}\n".format(indexes))
            return indexes

    def test_quo_and_qno(self):
        """Функция для проверки теста
        Кво и Кно

        Кво+Кно должно быть меньше 100%.

            Args:
                self.kv (array[int/float]): массив с данными кво для проверки
                self.kpp (array[int/float]): массив с данными кно для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        if self.check_input(self.kv, "kv", "test quo and qno") and \
                self.check_input(self.kpp, "kpp", "test quo and qno"):
            converted_kv = []
            converted_kpp = []
            result = True
            # перевод из процентов в доли при необходимости
            for val in self.kv:
                if val >= 1:
                    converted_kv.append(val / 100)
                else:
                    converted_kv.append(val)
            for val in self.kpp:
                if val >= 1:
                    converted_kpp.append(val / 100)
                else:
                    converted_kpp.append(val)

            for i in range(len(converted_kv)):
                if (converted_kv[i] + converted_kpp[i]) > 100:
                    result = False

            self.file.write(f"Test 'quo and qno': {result}."
                            f" Входной файл {self.file_name}. Дата выполнения {self.dt_now}\n")
            return result

    def test_correctness_of_p_sk_kp(self):
        """Функция для проверки теста
          ρ, Ск и Кп1

            Плотности, карбонатности и пористости в одноименных
            параллельных и перпендикулярных образцах должны быть одинаковыми

            Args:
                self.parallel (array[int/float]): массив с параллельными образцами
                self.parallel_density (array[int/float]): массив с плотностью для параллельных образцов
                self.parallel_carbonate (array[int/float]): массив с карбонатностью для параллельных образцов
                self.parallel_porosity (array[int/float]): массив с пористостью для параллельных образцов
                self.perpendicular (array[int/float]): массив с перпендикулярными образцами
                self.perpendicular_density (array[int/float]): массив с плотностью для перпендикулярных образцов
                self.perpendicular_carbonate (array[int/float]): массив с карбонатностью для перпендикулярных образцов
                self.perpendicular_porosity (array[int/float]): массив с пористостью для перпендикулярных образцов

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        if self.check_input(self.parallel, "parallel", "test correctness of p sk kp") and \
                self.check_input(self.parallel_density, "parallel_density", "test correctness of p sk kp") and \
                self.check_input(self.parallel_carbonate, "parallel_carbonate", "test correctness of p sk kp") and \
                self.check_input(self.parallel_porosity, "parallel_porosity", "test correctness of p sk kp") and \
                self.check_input(self.perpendicular, "perpendicular", "test correctness of p sk kp") and \
                self.check_input(self.perpendicular_density, "perpendicular_density",
                                 "test correctness of p sk kp") and \
                self.check_input(self.perpendicular_carbonate, "perpendicular_carbonate",
                                 "test correctness of p sk kp") and \
                self.check_input(self.perpendicular_porosity, "perpendicular_porosity", "test correctness of p sk kp"):
            # Создание словарей для хранения значений свойств по индексам
            parallel_properties = {}
            perpendicular_properties = {}

            # Заполнение словаря для первого массива
            for i in range(len(self.parallel)):
                parallel_properties[self.parallel[i]] = (
                    self.parallel_density[i], self.parallel_carbonate[i], self.parallel_porosity[i])

            # Заполнение словаря для второго массива
            for i in range(len(self.perpendicular)):
                perpendicular_properties[self.perpendicular[i]] = (
                    self.perpendicular_density[i], self.perpendicular_carbonate[i], self.perpendicular_porosity[i])

            # Сравнение значений свойств по одноименным индексам
            result = True
            for key in parallel_properties:
                if key in perpendicular_properties:
                    if parallel_properties[key] != perpendicular_properties[key]:
                        result = False

            self.file.write(f"Test 'quo and qno': {result}."
                            f" Входной файл {self.file_name}. Дата выполнения {self.dt_now}\n")
            return result

    def get_list_of_tests(self) -> list:
        """_summary_

        Returns:
            list: _description_
        """

        test_methods = [method for method in dir(self) if
                        callable(getattr(self, method)) and method.startswith("test")]
        return test_methods

    def start_tests(self, list_of_tests: list) -> list[Any]:
        """_summary_

        Args:
            list_of_tests (list): _description_
        """
        results = []
        for method_name in list_of_tests:
            method = getattr(self, method_name)
            results.append(method())
        return results

    def get_method_description(self, method_name: str) -> str:
        """Метод для получение описания теста по названию теста

         Args:
             method_name(str) - название теста
        Returns:
             str - описание теста
         """
        method = getattr(self, method_name, None)
        if method is not None:
            return method.__doc__
        else:
            return "Метод не найден."

    def generate_test_report(self) -> str:
        """_summary_

        Returns:
            str: _description_
        """
        self.file.close()
        new_filepath = self.file_path + "\\" + self.file_report_name  # Путь для сохранения файла на диске D:
        shutil.copy(self.file_report_name, new_filepath)  # Копирование файла
        os.remove(self.file_report_name)  # Удаление исходного файла
        return new_filepath
