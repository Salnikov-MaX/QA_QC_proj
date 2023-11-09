import numpy as np
from qa_qc_lib.tests.base_test import QA_QC_main


class QA_QC_kern(QA_QC_main):
    def __init__(self, file_path: str, depth=None,) -> None:
        super().__init__()
        self.depth = depth
        self.file_name = file_path.split('/')[-1]

    def __check_data(self, array, get_report=True):
        """
        Тест предназначен для проверки условия - все элементы массива должны быть числовыми.

            Args:
                self.data (array[T]): входной массив для проверки данных

            Returns:
                bool: результат выполнения теста
        """
        if not isinstance(array, np.ndarray):
            text = "Передан не массив"
            report_text = self.generate_report_text(text, 2)
            self.update_report(report_text)
            if get_report:
                print('\n' + report_text + self.delimeter)
            return False, array
        # сделано для отлавливания ситуаций np.array([None]), так как array.size выдаст 1,
        # но выпадет ошибка при обращении к индексу, lent проверяет случай None
        try:
            lent = array.size
            elem = array[0]
        except:
            text = "Передан пустой массив"
            report_text = self.generate_report_text(text, 2)
            self.update_report(report_text)
            if get_report:
                print('\n' + report_text + self.delimeter)
            return False, array

        string_indices = np.where(np.array(list(map(lambda x: isinstance(x, str), array))))
        if string_indices[0].size != 0:
            text = "Содержит не числовое значение"
            report_text = self.generate_report_text(text, 3)
            self.update_report(report_text)
            if get_report:
                print('\n' + report_text + self.delimeter)
            return False, string_indices
        for i in range(array.size):
            nan_mask = np.isnan(array)
            result_array = np.where(nan_mask, 1, 0)
            result = sum(result_array) == 0
            if not result:
                text = "Содержит nan"
                report_text = self.generate_report_text(text, 2)
                self.update_report(report_text)
                if get_report:
                    print('\n' + report_text + self.delimeter)
                return False, result_array
        return True,[]

    def test_monotony(self, get_report=True) -> dict:
        """
        Тест предназначен для проверки монотонности возрастания значения глубины

        Required data:
            Глубина отбора, м

        Args:
            self.core_sampling (array[int/float]): массив с местом отбора для проверки

        Returns:
            bool: результат выполнения теста
            file: запись результата теста для сохранения состояния
        """
        check_result, wrong = self.__check_data(self.depth, get_report)

        if check_result:
            result_mask = np.diff(self.depth) <= 0
            result_mask = np.insert(result_mask, 0, False)
            result = sum(result_mask) == 0

            if result:
                text = 'Глубина монотонно возрастает'
                report_text = self.generate_report_text(text, 1)
            else:
                text = 'Нарушена монотонность глубины'
                report_text = self.generate_report_text(text, 0)
            self.update_report(report_text)

            if get_report:
                print('\n' + report_text + self.delimeter)

            return {"data_availability": check_result,
                    "result": result,
                    "specification": {
                        "result_mask": result_mask,
                        "test_name": "test_monotony",
                        "param_name": "Глубина отбора, м",
                        "error_decr": "Нарушена монотонность"
                    }}

        else:
            return {"data_availability": check_result,
                    "result": False,
                    "specification": {
                        "result_mask": wrong,
                        "test_name": "test_monotony",
                        "param_name": "Глубина отбора, м",
                        "error_decr": "Нарушена монотонность"
                    }}
