import numpy as np
from qa_qc_lib.tests.base_test import QA_QC_main


class QA_QC_kern(QA_QC_main):
    def __init__(self, file_path: str, depth=None, ) -> None:
        super().__init__()
        self.depth = depth
        self.file_name = file_path.split('/')[-1]

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
        empty_wrong_text = "Передан пустой массив"
        not_array_wrong_text = "Передан не массив"
        not_int_wrong_text = "Содержит не числовое значение"
        has_nan_wrong_text = "Содержит nan"

        if not isinstance(array, np.ndarray):
            self.__generate_report(not_array_wrong_text, 2, get_report)
            return False, array, not_array_wrong_text
        # сделано для отлавливания ситуаций np.array([None]), так как array.size выдаст 1,
        # но выпадет ошибка при обращении к индексу, lent проверяет случай None
        try:
            lent = array.size
            elem = array[0]
        except:
            self.__generate_report(empty_wrong_text, 2, get_report)
            return False, array, empty_wrong_text

        string_indices = np.where(np.array(list(map(lambda x: isinstance(x, str), array))))
        if string_indices[0].size != 0:
            self.__generate_report(not_int_wrong_text, 2, get_report)
            return False, string_indices, not_int_wrong_text
        for i in range(array.size):
            nan_mask = np.isnan(array)
            result_array = np.where(nan_mask, 1, 0)
            result = sum(result_array) == 0
            if not result:
                self.__generate_report(has_nan_wrong_text, 2, get_report)
                return False, result_array, has_nan_wrong_text
        return True, [], ""

    def test_monotony(self, get_report=True) -> dict:
        """
        Тест предназначен для проверки монотонности возрастания значения глубины

        Required data:
            Глубина отбора, м

        Args:
            self.depth (array[int/float]): массив с местом отбора для проверки

        Returns:
            dict: Словарь где, data_availability - наличие данных, result - результат теста,
                  specification Словарь где ,result_mask - маска с результатом ,test_name - название теста ,
                  param_name - название параметра ,error_decr -краткое описание ошибки
        """

        wrong_text = "Нарушена монотонность глубины"
        accepted_text = "Глубина монотонно возрастает"
        check_result, wrong, check_text = self.__check_data(self.depth, get_report)

        if check_result:
            result_mask = np.diff(self.depth) <= 0
            result_mask = np.insert(result_mask, 0, False)
            result = sum(result_mask) == 0
            text = accepted_text if result else wrong_text
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
