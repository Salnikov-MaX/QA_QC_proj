import numpy as np
import pandas as pd

from qa_qc_lib.tests.base_test import QA_QC_main
from qa_qc_lib.tests.kern_tests.data_kern import DataKern
from qa_qc_lib.tests.kern_tests.kern_consts import KernConsts


class QA_QC_kern(QA_QC_main):
    def __init__(self, file_path: str, data_file_path=r"..\..\data\post_test_table.xlsx") -> None:
        super().__init__()
        self.upper_limit_poro = 0.476
        self.lower_limit_poro = 0
        self.__r2 = 0.7
        self.__alpha = 0.053
        self.file_name = file_path.split('/')[-1]
        self.data_kern = DataKern(data_file_path)
        self.consts = KernConsts()
        self.parameters_to_compare = {
            self.consts.kpr_abs: self.consts.kp_open,
            self.consts.poro_param: self.consts.kp_open,
            self.consts.kvo: self.consts.kp_open,
            self.consts.ads_density: self.consts.kp_open,
            self.consts.mms_density: self.consts.kp_open
        }

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

    def __check_data(self, array, get_report=True) -> tuple[bool, np.array, str]:
        """
        Тест предназначен для проверки условия - все элементы массива должны быть числовыми.

            Args:
                array (array[T]): входной массив для проверки данных

            Returns:
                bool: результат выполнения теста
                np.array: маска с ошибочными значениями
                str: описание ошибки
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

    def __generate_returns_dict(self, data_availability, result, result_masks, error_decr, well_name,
                                md, test_name, param_name, r2=None) -> dict:
        """
        Args:
            data_availability(bool): наличие данных
            result(bool): результат теста
            result_mask(np.ndarray[bool]): маска с результатом
            error_decr(string): описание ошибки
            well_name(string): название скважины
            md(np.ndarray[int/float]): массив с глубинами
            test_name(string): название теста
            param_name(array[string]): массив с названиями параметров

        Returns:
            dict: Словарь, specification cловарь где ,result_mask - маска с результатом ,test_name - название теста ,
                  param_name - название параметра ,error_decr -краткое описание ошибки

        """
        specification = {
            "test_name": test_name,
            "error_decr": error_decr,
            "well_name": str(well_name),
            "MD": md.tolist()
        }
        for index, param in enumerate(param_name):
            key_name = f"param_name_{index + 1}"
            specification[key_name] = param

        specification["result_masks"] = result_masks
        if r2 is not None:
            specification["r2"] = r2
        return {
            "data_availability": data_availability,
            "result": bool(result),
            "specification": specification
        }

    def __clean_data(self, nodes):
        """
        Метод для получение данных через data_kern
        Args:
            param(array[string]): массив с названиями параметров, которые необходимо получить

        Returns:
            clear_df(dataframe): dataframe с требуемыми параметрами очищенный от nan
            index(np.ndarray(int): индексы данных по глубине после удаления nan
            well_name(string): название скважины
            md(np.ndarray(int): массив с глубинами для которых взяты данные
        """
        data_dict = {"MD": [], "well_name": []}
        for node in nodes:
            data_dict["MD"].extend(node.MD)
            data_dict["well_name"].extend([node.well_name] * len(node.MD))
            data_dict[node.name] = data_dict.get(node.name, []) + node.data

        df = pd.DataFrame(data_dict)
        columns_to_clear = ["MD", "well_name"] + [node.name for node in nodes]
        clear_df = df[columns_to_clear].dropna()
        index = clear_df.index
        well_name = clear_df["well_name"].iloc[0]
        md = clear_df["MD"].values
        return clear_df, index, well_name, md

    def __zero_one_interval_check(self, array) -> tuple[np.array, bool]:
        """
        Функция для проверки нахождения данных в интервале либо от 0 до 1,

            Args:
                self.array (array[int/float]): массив с данными для проверки

            Returns:
                result_mask(np.ndarray[bool]): маска с выпадающими за интервал значениями
                result(bool):наличие ошибок в данных
        """
        result_mask = (array > 1) | (array <= 0)
        result = np.sum(result_mask) == 0
        return result_mask, result

    def test_zero_to_one_interval(self, node, get_report=True) -> dict:
        clear_df, index, well_name, md = self.__clean_data(nodes=[node])
        node=clear_df[node.name]
        check_result, wrong, check_text = self.__check_data(np.array(node), get_report)

        if check_result:
            result_mask, result = self.__zero_one_interval_check(node)
            text = self.consts.zero_one_interval_accepted if result else self.consts.zero_one_interval_wrong
            self.__generate_report(text, result, get_report)
            self.data_kern.dict_array.append({f"test_{node.name}": [node.name, text, result_mask, index]})

            return self.__generate_returns_dict(check_result, result, result_mask, text, well_name, md,
                                                f"test_{node.name}", [node.name])
        else:
            self.data_kern.dict_array.append({f"test_{node.name}": [node.name, check_text, wrong, index]})
            return self.__generate_returns_dict(check_result, False, wrong, check_text, well_name, md,
                                                f"test_{node.name}", [node.name])