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
        self.y = []
        self.x = []
        self.check_input()

    def __del__(self):
        self.file.close()

    """
    Тесты первого порядка 
    """

    def check_input(self):
        """Функция для проверки входных данных для тестов первого порядка
        Проверяет, что на вход подается не нулевой массив массив, содержащий только int и float

            Args:
                self.data (array[T]): входной массив для проверки данных

            Returns:
                bool: результат выполнения теста
        """

        if not isinstance(self.data, list):
            return False
        if len(self.data) == 0:
            return False
        for element in self.data:
            if not isinstance(element, (int, float)):
                return False
            if np.isnan(element):
                return False
        return True

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
        if all(0 <= x <= 1 for x in self.data):
            result = True
        else:
            result = False

        self.file.write("Test 'water_saturation': {}\n".format(result))
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

        if all(0 <= x <= 47.6 for x in self.data):
            result = True
        else:
            result = False
        self.file.write("Test 'porosity': {}\n".format(result))
        return result

    def test_permeability(self) -> bool:
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

        if all(x > 0 for x in self.data):
            result = True
        else:
            result = False
        self.file.write("Test 'permeability': {}\n".format(result))
        return result

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

        result = True
        for i in range(1, len(self.data)):
            if self.data[i] <= self.data[i - 1]:
                result = False
        self.file.write("Test 'monotony': {}\n".format(result))
        return result

    """
        Тесты второго порядка
    """

    def test_quo_kp_dependence(self) -> bool:
        """Функция для проверки тестов
        Зависимость Кво-Кп
        Зависимость Кп-плотность
        Зависимость Кво-Кпдин

        Необходимо построить линию тренда и проверить, что зависимость линейная по функции y=a*x+b, при этом a<0

            Args:
                self.x (array[int/float]): массив с данными кво/кп/кво для проверки
                self.y (array[int/float]): массив с данными кп/плотность/кпдин для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """

        coefficients = np.polyfit(self.data.x, self.data.y, 1)
        a = coefficients[0]
        result = True
        if a >= 0:
            result = False
        self.file.write("Test 'dependence quo kp': {}\n".format(result))
        return result

    def test_obblnas_kp_dependence(self) -> bool:
        """Функция для проверки теста
        Обплнас-Кп

        Необходимо построить линию тренда и проверить,зависимость по функции y=a1*x+b1, при этом a1<a2,
         где а2 - коэффициент из зависимости Минпл-Кп


            Args:
                self.x (array[int/float]): массив с данными кво для проверки
                self.y (array[int/float]): массив с данными кп для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        coeffs1 = np.polyfit(self.data.x1, self.data.y1, 1)
        a1, b1 = coeffs1[0], coeffs1[1]

        coeffs2 = np.polyfit(self.data.x2, self.data.y2, 1)
        a2, b2 = coeffs2[0], coeffs2[1]
        result = True
        if a1 >= a2:
            result = False

        self.file.write("Test 'dependence obblnas kp': {}\n".format(result))
        return result

    def test_minple_kp_dependence(self) -> bool:
        """Функция для проверки теста
        Минпл-Кп

        Необходимо построить линию тренда и проверить,зависимость по функции y=a1*x+b1, при этом a1<a2,
        где а1 - коэффициент из зависимости Обплнас-Кп

            Args:
                self.x (array[int/float]): массив с данными кво для проверки
                self.y (array[int/float]): массив с данными кп для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """

        coeffs1 = np.polyfit(self.data.x1, self.data.y1, 1)
        a1, b1 = coeffs1[0], coeffs1[1]

        coeffs2 = np.polyfit(self.data.x2, self.data.x2, 1)
        a2, b2 = coeffs2[0], coeffs2[1]
        result = True
        if a1 <= a2:
            result = False

        self.file.write("Test 'dependence minple kp': {}\n".format(result))
        return result

    def test_kpf_pdin_dependence(self) -> bool:
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

        coefficients = np.polyfit(self.data.x, self.data.y, 1)
        a = coefficients[0]
        b = coefficients[1]
        result = True
        if a <= 0 or b <= 0:
            result = False
        self.file.write("Test 'dependence kpf pdin': {}\n".format(result))
        return result

    def test_kpff_kp_dependence(self) -> bool:
        """Функция для проверки тестов
        Зависимость Кпэф-Кп
        Зависимость Кп-Кпдин

        Необходимо построить линию тренда и проверить,что зависимость линейная по функции y=a*x+b, при этом a>0, b<0

            Args:
                self.x (array[int/float]): массив с данными кпэф,кп для проверки
                self.y (array[int/float]): массив с данными кп,кпдин для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """

        coefficients = np.polyfit(self.data.x, self.data.y, 1)
        a = coefficients[0]
        b = coefficients[1]
        result = True
        if a <= 0 or b >= 0:
            result = False

        self.file.write("Test 'dependence kpff kp': {}\n".format(result))
        return result

    def test_dependence_kpc_kp(self) -> bool:
        """Функция для проверки тестов
        Зависимость Кпр-Кп
        Зависимость Кпр-Кпдин

        Необходимо построить линию тренда и проверить,что  зависимость по функции y=a*exp(b*x) при этом b>0

            Args:
                self.x (array[int/float]): массив с данными кпр,кпр для проверки
                self.y (array[int/float]): массив с данными кп,кпдин для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
                """

        coefficients = np.polyfit(self.data.x, np.log(self.data.y), 1)
        b = coefficients[1]
        result = True
        if b <= 0:
            result = False

        self.file.write("Test 'dependence kpc kp': {}\n".format(result))
        return result

    def test_dependence_quo_qp(self) -> bool:
        """Функция для проверки тестов
        Зависимость Кво-Кпр

        Необходимо построить линию тренда и проверить,что зависимость по функции y=a*ln(x)+b при этом a>0

            Args:
                self.x (array[int/float]): массив с данными кво для проверки
                self.y (array[int/float]): массив с данными кпр для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
                """

        coefficients = np.polyfit(self.data.x, np.exp(self.data.y), 1)
        a, b = coefficients[0], coefficients[1]
        result = True
        if a <= 0:
            result = False
        self.file.write("Test 'dependence quo qp': {}\n".format(result))
        return result

    def test_rn_kv_dependencies(self) -> bool:
        """Функция для проверки тестов
        Зависимости Рн-Кв

        Необходимо построить линию тренда и проверить,что зависимость по функции y=b/(kв^n) при этом 1,1<n<5

            Args:
                self.x (array[int/float]): массив с данными рн для проверки
                self.y (array[int/float]): массив с данными кв для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """

        coefficients = np.polyfit(np.log(self.data.x), np.log(self.data.y), 1)
        b, n = np.exp(coefficients[1]), coefficients[0]
        result = True
        if 1.1 >= n or n >= 5:
            result = False

        self.file.write("Test 'dependence rn kv': {}\n".format(result))
        return result

    def test_rn_kn_dependencies(self) -> bool:
        """Функция для проверки тестов
        Зависимость Кпр-Кп
        Зависимость Кпр-Кпдин

        Необходимо построить линию тренда и проверить,что  зависимость по функции y=a*exp(b*x) при этом b>0

            Args:
                self.x (array[int/float]): массив с данными кпр,кпр для проверки
                self.y (array[int/float]): массив с данными кп,кпдин для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """
        """

        
        """
        coefficients = np.polyfit(np.log(self.data.x), -np.log(self.data.y), 1)
        a, m = np.exp(-coefficients[1]), coefficients[0]
        result = True
        if 1.1 >= m or m >= 3.8 or 0 >= a or a >= 2.5:
            result = False

        self.file.write("Test 'dependence rn kn ': {}\n".format(result))
        return result

    def test_general_dependency_checking(self) -> bool:
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
                self.x (array[int/float]): массив с данными для проверки
                self.y (array[int/float]): массив с данными для проверки

            Returns:
                bool: результат выполнения теста
                file: запись результата теста для сохранения состояния
        """

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

        result = True
        for i in range(len(self.roof)):
            if self.sole[i] < self.roof[i]:
                result = False
        self.file.write("Test 'coring depths first': {}\n".format(result))
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
        result = True
        for i in range(len(self.roof)):
            if self.sole[i] - self.roof[i] > self.takeout[0]:
                result = False
        self.file.write("Test 'coring depths second': {}\n".format(result))
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

        result = True
        for i in range(len(self.intervals)):
            interval_length = max(self.intervals[i]) - min(self.intervals[i])
            displacement_percent_calculated = (self.outreach_in_meters[i] / interval_length) * 100
            if abs(self.percentage[i] - displacement_percent_calculated) > 0.001:
                result = False

        self.file.write("Test 'coring depths third': {}\n".format(result))
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

        result = True
        for i in range(len(self.sampling_depth)):
            if self.core_sampling[i] > self.sampling_depth[i]:
                result = False

        self.file.write("Test 'coring depths four': {}\n".format(result))
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

        self.file.write("Test 'data tampering': {}\n".format(result))
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

        self.file.write("Test 'estimated values in core data': {}\n".format(result))
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

        result = True
        for i in range(len(self.kp_pov)):
            if abs(self.kp_pov[i] - self.kp_plast[i]) > 2:
                result = False

        self.file.write("Test 'kp in surface and reservoir conditions': {}\n".format(result))
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

        converted_kv = []
        converted_kpp = []
        result = True
        # перевод из долей в проценты при необходимости
        for val in self.kv:
            if val <= 1:
                converted_kv.append(val * 100)
            else:
                converted_kv.append(val)
        for val in self.kpp:
            if val <= 1:
                converted_kpp.append(val * 100)
            else:
                converted_kpp.append(val)

        for i in range(len(converted_kv)):
            if (converted_kv[i] + converted_kpp[i]) > 100:
                result = False

        self.file.write("Test 'quo and qno': {}\n".format(result))
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

        self.file.write("Test 'quo and qno': {}\n".format(result))
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

qa=QA_QC_kern()
print(qa.get_list_of_tests())