import os
import shutil
import numpy as np
from scipy.stats import t
from sklearn.metrics import r2_score


class QA_QC_kern:
    def __init__(self, filename="test_report.txt") -> None:
        """_summary_

        Args:
            data (str): _description_
        """
        self.filename = filename
        self.file = open(self.filename, "w")
        self.y = [
            0.052,
            0.076,
            0.018,
            0.119,
            0.138,
            0.156,
            0.16,
            0.165,
            0.138,
            0.159,
            0.133,
            0.159,
            0.147,
            0.146,
            0.084,
            0.097,
            0.125,
            0.135,
            90
        ]
        self.x = [
            0.052,
            0.076,
            0.018,
            0.119,
            0.138,
            0.156,
            0.16,
            0.165,
            0.138,
            0.159,
            0.133,
            0.159,
            0.147,
            0.146,
            0.084,
            0.097,
            0.125,
            0.135,
            90
        ]
        self.check_input()
        pass

    def check_input(self):
        if not isinstance(self, list):
            return False
        if len(self) == 0:
            return False
        for element in self:
            if not isinstance(element, (int, float)):
                return False

    def __del__(self):
        self.file.close()

    def test_water_saturation(self) -> bool:

        if all(0 <= x <= 1 for x in self.data):
            result = True
        else:
            result = False

        self.file.write("Test 'water_saturation': {}\n".format(result))
        return result

    def test_porosity(self) -> bool:
        if all(0 <= x <= 47.6 for x in self.data):
            result = True
        else:
            result = False
        self.file.write("Test 'porosity': {}\n".format(result))
        return result

    def test_permeability(self) -> bool:
        if all(x > 0 for x in self.data):
            result = True
        else:
            result = False
        self.file.write("Test 'permeability': {}\n".format(result))
        return result

    def test_monotony(self) -> bool:
        result = True
        for i in range(1, len(self.data)):
            if self.data[i] <= self.data[i - 1]:
                result = False
        self.file.write("Test 'monotony': {}\n".format(result))
        return result

    def test_quo_quo_dependence(self):
        coefficients = np.polyfit(self.data.x, self.data.y, 1)
        a = coefficients[0]
        b = coefficients[1]
        result = True
        if a >= 0:
            result = False
        self.file.write("Test 'dependence quo quo': {}\n".format(result))
        return result

    def test_obblnas_kp_dependence(self):
        coeffs1 = np.polyfit(self.data.x1, self.data.y1, 1)
        a1, b1 = coeffs1[0], coeffs1[1]

        coeffs2 = np.polyfit(self.data.x2, self.data.y2, 1)
        a2, b2 = coeffs2[0], coeffs2[1]
        result = True
        if a1 >= a2:
            result = False

        self.file.write("Test 'dependence obblnas kp': {}\n".format(result))
        return result

    def test_minple_kp_dependence(self):
        coeffs1 = np.polyfit(self.data.x1, self.data.y1, 1)
        a1, b1 = coeffs1[0], coeffs1[1]

        coeffs2 = np.polyfit(self.data.x2, self.data.x2, 1)
        a2, b2 = coeffs2[0], coeffs2[1]
        result = True
        if a1 <= a2:
            result = False

        self.file.write("Test 'dependence minple kp': {}\n".format(result))
        return result

    def test_kpf_pdin_dependence(self):
        coefficients = np.polyfit(self.data.x, self.data.y, 1)
        a = coefficients[0]
        b = coefficients[1]
        result = True
        if a <= 0 or b <= 0:
            result = False
        self.file.write("Test 'dependence kpf pdin': {}\n".format(result))
        return result

    def test_kpff_kp_dependence(self):
        coefficients = np.polyfit(self.data.x, self.data.y, 1)
        a = coefficients[0]
        b = coefficients[1]
        result = True
        if a <= 0 or b >= 0:
            result = False

        self.file.write("Test 'dependence kpff kp': {}\n".format(result))
        return result

    def test_dependence_kpc_kp(self):
        coefficients = np.polyfit(self.data.x, np.log(self.data.y), 1)
        a = coefficients[0]
        b = coefficients[1]
        result = True
        if b <= 0:
            result = False

        self.file.write("Test 'dependence kpc kp': {}\n".format(result))
        return result

    def test_dependence_quo_qp(self):
        coefficients = np.polyfit(self.data.x, np.exp(self.data.y), 1)
        a, b = coefficients[0], coefficients[1]
        result = True
        if a <= 0:
            result = False
        self.file.write("Test 'dependence quo qp': {}\n".format(result))
        return result

    def test_rn_kv_dependencies(self):
        coefficients = np.polyfit(np.log(self.data.x), np.log(self.data.y), 1)
        b, n = np.exp(coefficients[1]), coefficients[0]
        result = True
        if 1.1 >= n or n >= 5:
            result = False

        self.file.write("Test 'dependence rn kv': {}\n".format(result))
        return result

    def test_rn_kn_dependencies(self):
        coefficients = np.polyfit(np.log(self.data.x), -np.log(self.data.y), 1)
        a, m = np.exp(-coefficients[1]), coefficients[0]
        result = True
        if 1.1 >= m or m >= 3.8 or 0 >= a or a >= 2.5:
            result = False

        self.file.write("Test 'dependence rn kn ': {}\n".format(result))
        return result

    def general_dependency_checking(self):
        alpha = 0.05  # Уровень значимости
        n = len(self.x)
        dof = n - 2  # Число степеней свободы для распределения Стьюдента

        residuals = self.y - np.polyval(np.polyfit(self.x, self.y, 1), self.x)
        std_error = np.sqrt(np.sum(residuals ** 2) / dof)

        t_critical = t.ppf(1 - alpha / 2, dof)
        upper_limit = np.polyval(np.polyfit(self.x, self.y, 1), self.x) + t_critical * std_error
        lower_limit = np.polyval(np.polyfit(self.x, self.y, 1), self.x) - t_critical * std_error

        x_filtered = []
        y_filtered = []

        for i in range(n):
            if lower_limit[i] <= self.y[i] <= upper_limit[i]:
                x_filtered.append(self.x[i])
                y_filtered.append(self.y[i])

        # Аппроксимация линии тренда на отфильтрованных данных
        coeffs = np.polyfit(x_filtered, y_filtered, 1)
        trend_line = np.polyval(coeffs, x_filtered)

        # Вычисление R2 score
        r2 = r2_score(y_filtered, trend_line)

        # Проверка условия R2 score и удаление точек при несоответствии
        while r2 < 0.7 and len(x_filtered) > 0.9 * n:
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
            self.file.write("Test 'general dependency checking': {}\n".format(result))
            return result
        else:
            result = False
            self.file.write("Test 'general dependency checking': {}\n".format(result))
            return result

    def get_list_of_tests(self) -> list:
        """_summary_

        Returns:
            list: _description_
        """

        test_methods = [method for method in dir(self) if
                        callable(getattr(self, method)) and method.startswith("test")]
        return test_methods

    def start_tests(self, list_of_tests: list) -> None:
        """_summary_

        Args:
            list_of_tests (list): _description_
        """
        for method_name in list_of_tests:
            method = getattr(self, method_name)
            method()
        pass

    def generate_test_report(self) -> str:
        """_summary_

        Returns:
            str: _description_
        """
        self.file.close()
        new_filepath = "D:\\" + self.filename  # Путь для сохранения файла на диске D:
        shutil.copy(self.filename, new_filepath)  # Копирование файла
        os.remove(self.filename)  # Удаление исходного файла
        return new_filepath
