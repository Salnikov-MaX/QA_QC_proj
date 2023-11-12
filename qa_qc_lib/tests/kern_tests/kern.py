import numpy as np
from scipy.stats import t
from sklearn.metrics import r2_score
from qa_qc_lib.tests.base_test import QA_QC_main
from qa_qc_lib.tests.kern_tests.kern_consts import KernConsts


class QA_QC_kern(QA_QC_main):
    def __init__(self, file_path: str, depth=None, porosity_abs=None, porosity_open=None, sw_residual=None) -> None:
        super().__init__()
        self.sw_residual = sw_residual
        self.porosity_open = porosity_open
        self.depth = depth
        self.__r2 = 0.7
        self.__alpha = 0.053
        self.file_name = file_path.split('/')[-1]
        self.consts = KernConsts()
        self.porosity_abs = porosity_abs

    def __generate_report(self, text, status, get_report):
        """
        Метод группирующий основные операции по генерации отчета

        Args:
        text(string): краткое описание результата теста
        status(int): статус теста
        get_report(bool): флаг для получения отчета

        Returns:
        _type_: _description_
        """
        report_text = self.generate_report_text(text, status)
        self.update_report(report_text)
        if get_report:
            print(report_text)

    def __check_data(self, array, get_report=True):
        """
        Тест предназначен для проверки условия - все элементы массива должны быть числовыми.

            Args:
                self.data (array[T]): входной массив для проверки данных

            Returns:
                bool: результат выполнения теста
        """

        if not isinstance(array, np.ndarray):
            self.__generate_report(self.consts.check_data_not_array_wrong, 2, get_report)
            return False, array, self.consts.check_data_not_array_wrong
        # сделано для отлавливания ситуаций np.array([None]), так как array.size выдаст 1,
        # но выпадет ошибка при обращении к индексу, lent проверяет случай None
        try:
            lent = array.size
            elem = array[0]
        except:
            self.__generate_report(self.consts.check_data_empty_wrong, 2, get_report)
            return False, array, self.consts.check_data_empty_wrong

        string_indices = np.where(np.array(list(map(lambda x: isinstance(x, str), array))))
        if string_indices[0].size != 0:
            self.__generate_report(self.consts.check_data_not_int_wrong, 2, get_report)
            return False, string_indices, self.consts.check_data_not_int_wrong
        for i in range(array.size):
            nan_mask = np.isnan(array)
            result_array = np.where(nan_mask, 1, 0)
            result = sum(result_array) == 0
            if not result:
                self.__generate_report(self.consts.check_data_has_nan_wrong, 2, get_report)
                return False, result_array, self.consts.check_data_has_nan_wrong
        return True, [], ""

    def test_monotony(self, get_report=True) -> dict:
        """
        Тест предназначен для проверки монотонности возрастания значения глубины

        Required data:
            Глубина отбора, м

        Args:
            self.depth (array[int/float]): массив с местом отбора для проверки

        Returns:
            dict: Словарь, specification cловарь где ,result_mask - маска с результатом ,test_name - название теста ,
                  param_name - название параметра ,error_decr -краткое описание ошибки
        """

        check_result, wrong, check_text = self.__check_data(self.depth, get_report)

        if check_result:
            result_mask = np.diff(self.depth) <= 0
            result_mask = np.insert(result_mask, 0, False)
            result = sum(result_mask) == 0
            text = self.consts.monotony_accepted if result else self.consts.monotony_wrong
            self.__generate_report(text, result, get_report)

            return {"data_availability": check_result,
                    "result": result,
                    "specification": {
                        "result_mask": result_mask,
                        "test_name": "test_monotony",
                        "param_name": "Глубина отбора, м",
                        "error_decr": text
                    }}

        else:
            return {"data_availability": check_result,
                    "result": False,
                    "specification": {
                        "result_mask": wrong,
                        "test_name": "test_monotony",
                        "param_name": "Глубина отбора, м",
                        "error_decr": check_text
                    }}

    def __check_poro_intervals(self, porosity):
        """
        Функция для проверки нахождения данных в интервале либо от 0 до 0.476,
        в случае, если пористость дана в долях, либо в интерваое от 0 до 47.6,
        если пористость дана в процентах

        Args:
            self.porosity (array[int/float]): массив с пористостью для проверки

        Returns:
            result_mask(np.ndarray[bool]): маска с выпадающими за интервал занчениями
            result(bool):наличие ошибок в пористости
            """
        lower_limit, upper_limit = (0, 0.476) if np.mean((0 <= porosity) &
                                                         (porosity <= 1)) > 0.5 else (0, 47.6)

        result_mask = (porosity > upper_limit) | (porosity < lower_limit)
        result = sum(result_mask) == 0
        return result_mask, result

    def test_porosity_open(self, get_report=True) -> dict:
        """
        Тест предназначен для проверки физичности данных.
        В данном тесте проверяется соответствие интервалу (0 ; 47,6]

        Required data:
            Кп_откр

        Args:
            self.porosity_open (array[int/float]): массив с кп_откр для проверки

        Returns:
            dict: Словарь, specification cловарь где ,result_mask - маска с результатом ,test_name - название теста ,
                      param_name - название параметра ,error_decr -краткое описание ошибки
            """

        check_result, wrong, check_text = self.__check_data(self.porosity_open, get_report)

        if check_result:
            result_mask, result = self.__check_poro_intervals(self.porosity_open)
            text = self.consts.porosity_interval_accepted if result else self.consts.porosity_interval_wrong
            self.__generate_report(text, result, get_report)

            return {"data_availability": check_result,
                    "result": result,
                    "specification": {
                        "result_mask": result_mask,
                        "test_name": "test_porosity_open",
                        "param_name": "Кп_откр",
                        "error_decr": text
                    }}

        else:
            return {"data_availability": check_result,
                    "result": False,
                    "specification": {
                        "result_mask": wrong,
                        "test_name": "test_porosity_open",
                        "param_name": "Кп_откр",
                        "error_decr": check_text
                    }}

    def test_porosity_abs(self, get_report=True) -> dict:
        """
        Тест предназначен для проверки физичности данных.
        В данном тесте проверяется соответствие интервалу (0 ; 47,6]

        Required data:
            Кп_абс

        Args:
            self.porosity_abs (array[int/float]): массив с кп_абс для проверки

        Returns:
            dict: Словарь, specification cловарь где ,result_mask - маска с результатом ,test_name - название теста ,
                      param_name - название параметра ,error_decr -краткое описание ошибки
            """

        check_result, wrong, check_text = self.__check_data(self.porosity_abs, get_report)

        if check_result:
            result_mask, result = self.__check_poro_intervals(self.porosity_abs)
            text = self.consts.porosity_interval_accepted if result else self.consts.porosity_interval_wrong
            self.__generate_report(text, result, get_report)

            return {"data_availability": check_result,
                    "result": result,
                    "specification": {
                        "result_mask": result_mask,
                        "test_name": "test_porosity_abs",
                        "param_name": "Кп_абс",
                        "error_decr": text
                    }}

        else:
            return {"data_availability": check_result,
                    "result": False,
                    "specification": {
                        "result_mask": wrong,
                        "test_name": "test_porosity_abs",
                        "param_name": "Кп_абс",
                        "error_decr": check_text
                    }}

    def test_general_dependency_checking(self, x, y, get_report=True):
        """
                Тест предназначен для оценки дисперсии входных данных.
                Он проводится по следующему алгоритму: изначально,
                используя статистические методы, детектируются и удаляются
                выбросные точки, затем полученное облако точек  аппроксимируется
                и считается коэффициент детерминации R2. Если его значение больше
                0.7, то тест считается пройденным. Если значение меньше 0.7, то
                точки сортируются по удаленности от линии тренда, и запускается
                цикл, за одну итерацию которого удаляется самая отдаленная от
                линии аппроксимации точка, и считается R2, если значение больше
                0.7, и удалено менее 10% точек, то тест пройден, иначе - нет.

                    Args:
                        x (array[int/float]): массив с данными для проверки
                        y (array[int/float]): массив с данными для проверки

                    Returns:
                        dict: Словарь с результатом теста, коэффиентом r2
                """

        result = False

        n = len(x)
        dof = n - 2  # Число степеней свободы для распределения Стьюдента
        x = list(x)
        y = list(y)
        residuals = y - np.polyval(np.polyfit(x, y, 1), x)
        std_error = np.sqrt(np.sum(residuals ** 2) / dof)

        t_critical = t.ppf(1 - self.__alpha / 2, dof)
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
        # первая проверка, так как мы можем не попасть дальше в цикл
        if r2 >= self.__r2:
            result = True
        # Проверка условия R2 score и удаление точек при несоответствии
        while r2 < self.__r2 and len(x_filtered) > 0.9 * n:
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
            if r2 >= self.__r2:
                result = True
                break

        self.__generate_report(self.consts.general_dependency_checking_accepted if result
                               else self.consts.general_dependency_checking_wrong, result, get_report)
        return {
            "result": result,
            "specification": {
                "r2": r2
            }}
