import numpy as np
from pandas import DataFrame
from scipy.stats import t
from sklearn.metrics import r2_score
from qa_qc_lib.tests.base_test import QA_QC_main
from qa_qc_lib.tests.kern_tests.data_kern import DataKern
from qa_qc_lib.tests.kern_tests.kern_consts import KernConsts
from qa_qc_lib.tools.math_tools import linear_dependence_function, exponential_function
from qa_qc_lib.tools.kern_tools import linear_function_visualization
from qa_qc_lib.tools.kern_tools import dropdown_search


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

    def __greater_than_zero_check(self, array) -> tuple[np.array, bool]:
        """
        Функция для проверки, что данные больше 0,

        Args:
            self.array (np.ndarray[int/float]): массив с данными для проверки

        Returns:
            result_mask(np.ndarray[bool]): маска с выпадающими за интервал значениями
            result(bool):наличие ошибок в данных
        """
        result_mask = (array <= 0)
        result = np.sum(result_mask) == 0
        return result_mask, result

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

    def __get_data_from_data_kern(self, param, filters) -> tuple[DataFrame, np.array, str, np.array]:
        """
        Метод для получение данных через data_kern
        Args:
            param(array[string]): массив с названиями параметров, которые необходимо получить
            filters(array[dic]): применяемые фильтры в формате [{"name":str,"value":str||int,
                                                                "operation"(np.ndarray[string]):[=, !=, >, <, >=, <=]}]

        Returns:
            clear_df(dataframe): dataframe с требуемыми параметрами очищенный от nan
            index(np.ndarray(int): индексы данных по глубине после удаления nan
            well_name(string): название скважины
            md(np.ndarray(int): массив с глубинами для которых взяты данные
        """
        columns = [self.consts.well, self.consts.md] + param
        df = self.data_kern.get_attributes(column_names=columns, filters=filters)
        columns_to_clear = [self.consts.md] + param
        clear_df = df[columns_to_clear].dropna()
        index = clear_df.index
        well_name = df[self.consts.well].iloc[0]
        md = clear_df[self.consts.md]
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

    def __check_poro_intervals(self, porosity) -> tuple[np.array, bool]:
        """
        Функция для проверки нахождения данных в интервале от 0 до 0.476.

        Args:
            porosity (np.ndarray[int/float]): массив с пористостью для проверки

        Returns:
            result_mask(np.ndarray[bool]): маска с выпадающими за интервал значениями
            result(bool):наличие ошибок в пористости
            """
        lower_limit, upper_limit = 0, 0.476

        result_mask = (porosity > upper_limit) | (porosity <= lower_limit)
        result = sum(result_mask) == 0
        return result_mask, result

    def __main_porosity_test(self, poro_name, test_name, get_report, filters) -> dict:
        """
        Главный метод для проверки на физичность любого вида пористости.
        Args:
            poro_name: название передаваемой пористости
            test_name: название теста, где вызван метод
            get_report: флаг для получения отчета
            filters(array[dic]): применяемые фильтры в формате [{"name":str,"value":str||int,
                                                                "operation"(np.ndarray[string]):[=, !=, >, <, >=, <=]}]

        Returns:
            dict: Словарь, specification cловарь где ,result_mask - маска с результатом ,test_name - название теста ,
                      param_name - название параметра ,error_decr -краткое описание ошибки,well_name- название скважины,
                      MD - массив с глубинами
        """
        clear_df, index, well_name, md = self.__get_data_from_data_kern(param=[poro_name], filters=filters)
        porosity_open = np.array(clear_df[poro_name])
        check_result, wrong, check_text = self.__check_data(porosity_open, get_report)

        if check_result:
            result_mask, result = self.__check_poro_intervals(porosity_open)
            text = self.consts.porosity_interval_accepted if result else self.consts.porosity_interval_wrong
            self.__generate_report(text, result, get_report)
            self.data_kern.mark_errors(poro_name, test_name, text, result_mask, index)

            return self.__generate_returns_dict(check_result, result, result_mask, text, well_name, md,
                                                test_name, [poro_name])
        else:
            self.data_kern.mark_errors(poro_name, test_name, check_text, wrong, index)
            return self.__generate_returns_dict(check_result, False, wrong, check_text, well_name, md,
                                                test_name, [poro_name])

    def __main_zero_to_one_interval_test(self, param_name, test_name, get_report, filters) -> dict:
        """
        Главный класс для проверки на физичность любого вида пористости.
        Args:
            param_name: название передаваемого параметра
            test_name: название теста, где вызван метод
            get_report: флаг для получения отчета
            filters(array[dic]): применяемые фильтры в формате [{"name":str,"value":str||int,
                                                                "operation"(np.ndarray[string]):[=, !=, >, <, >=, <=]}]

        Returns:
            dict: Словарь, specification cловарь где ,result_mask - маска с результатом ,test_name - название теста ,
                      param_name - название параметра ,error_decr -краткое описание ошибки,well_name- название скважины,
                      MD - массив с глубинами
        """
        clear_df, index, well_name, md = self.__get_data_from_data_kern(param=[param_name], filters=filters)
        param = np.array(clear_df[param_name])
        check_result, wrong, check_text = self.__check_data(param, get_report)

        if check_result:
            result_mask, result = self.__zero_one_interval_check(param_name)
            text = self.consts.zero_one_interval_accepted if result else self.consts.zero_one_interval_wrong
            self.__generate_report(text, result, get_report)
            self.data_kern.mark_errors(param_name, test_name, text, result_mask, index)

            return self.__generate_returns_dict(check_result, result, result_mask, text, well_name, md,
                                                test_name, [param_name])
        else:
            self.data_kern.mark_errors(param_name, test_name, check_text, wrong, index)
            return self.__generate_returns_dict(check_result, False, wrong, check_text, well_name, md,
                                                test_name, [param_name])

    def __main_greater_than_zero(self, param_name, test_name, get_report, filters) -> dict:
        """
        Главный метод для проверки, что все значения переданного параметра больше 0.

        Args:
            param_name(string): название параметра для проверки
            test_name(string): название теста, где вызван метод
            get_report(bool): флаг для получения отчета
            filters(array[dic]): применяемые фильтры в формате [{"name":str,"value":str||int,
                                                                "operation"(np.ndarray[string]):[=, !=, >, <, >=, <=]}]

        Returns:
            dict: Словарь, specification cловарь где ,result_mask - маска с результатом ,test_name - название теста ,
                      param_name - название параметра ,error_decr -краткое описание ошибки,well_name- название скважины,
                      MD - массив с глубинами
        """
        clear_df, index, well_name, md = self.__get_data_from_data_kern(param=[param_name], filters=filters)
        param = np.array(clear_df[param_name])
        check_result, wrong, check_text = self.__check_data(param, get_report)

        if check_result:
            result_mask, result = self.__greater_than_zero_check(param)
            text = self.consts.greater_than_zero_accepted if result else self.consts.greater_than_zero_wrong
            self.__generate_report(text, result, get_report)
            self.data_kern.mark_errors(param_name, test_name, text, result_mask, index)

            return self.__generate_returns_dict(check_result, result, result_mask, text, well_name, md,
                                                test_name, [param_name])
        else:
            return self.__generate_returns_dict(check_result, False, wrong, check_text, well_name, md,
                                                test_name, [param_name])

    def __check_vp_vs(self, vp_or_vs_param, test_name, get_report, filters) -> dict:
        """
        Функция для проверки нахождения данных в интервале от 0.3 до 10

        Args:
            vp_or_vs_param (np.ndarray[int/float]): массив со скоростью для проверки
            test_name: название теста, где вызван метод
            get_report: флаг для получения отчета
            filters(array[dic]): применяемые фильтры в формате [{"name":str,"value":str||int,
                                                                "operation"(np.ndarray[string]):[=, !=, >, <, >=, <=]}]

        Returns:
            dict: Словарь, specification cловарь где ,result_mask - маска с результатом ,test_name - название теста ,
                      param_name - название параметра ,error_decr -краткое описание ошибки,well_name- название скважины,
                      MD - массив с глубинами
        """
        clear_df, index, well_name, md = self.__get_data_from_data_kern(param=[vp_or_vs_param], filters=filters)
        vp_or_vs = np.array(clear_df[vp_or_vs_param])
        check_result, wrong, check_text = self.__check_data(vp_or_vs, get_report)

        if check_result:
            result_mask = (vp_or_vs >= 10) | (vp_or_vs <= 0.3)
            result = sum(result_mask) == 0
            text = self.consts.vp_vs_accepted if result else self.consts.vp_vs_wrong
            self.__generate_report(text, result, get_report)
            self.data_kern.mark_errors(vp_or_vs_param, test_name, text, result_mask, index)

            return self.__generate_returns_dict(check_result, result, result_mask, text, well_name, md,
                                                test_name, [vp_or_vs])
        else:
            self.data_kern.mark_errors(vp_or_vs_param, test_name, check_text, wrong, index)
            return self.__generate_returns_dict(check_result, False, wrong, check_text, well_name, md,
                                                test_name, [vp_or_vs])

    def test_monotony(self, get_report=True, filters=None) -> dict:
        """
        Тест предназначен для проверки монотонности возрастания значения глубины

        Required data:
            Глубина отбора, м

        Args:
            Глубина отбора, м (np.ndarray[int/float]): массив с местом отбора для проверки из переданной таблицы

        Returns:
            dict: Словарь, specification cловарь где ,result_mask - маска с результатом ,test_name - название теста ,
                      param_name - название параметра ,error_decr -краткое описание ошибки,well_name- название скважины,
                      MD - массив с глубинами
        """

        clear_df, index, well_name, md = self.__get_data_from_data_kern(param=[], filters=filters)
        depth = np.array(clear_df[self.consts.md])
        check_result, wrong, check_text = self.__check_data(depth, get_report)

        if check_result:
            result_mask = np.diff(depth) <= 0
            result_mask = np.insert(result_mask, 0, False)
            result = sum(result_mask) == 0
            text = self.consts.monotony_accepted if result else self.consts.monotony_wrong
            self.__generate_report(text, result, get_report)

            self.data_kern.mark_errors(self.consts.md, "test_monotony", text, result_mask, index)
            return self.__generate_returns_dict(check_result, result, result_mask, text, well_name, md,
                                                "test_monotony", [self.consts.md])

        else:
            self.data_kern.mark_errors(self.consts.md, "test_monotony", check_text, wrong, index)
            return self.__generate_returns_dict(check_result, False, wrong, check_text, well_name, md,
                                                "test_monotony", [self.consts.md])

    def test_vp(self, get_report=True, filters=None) -> dict:
        """
            Тест предназначен для проверки физичности данных.
            В данном тесте проверяется соответствие интервалу 0.3<Vp<10 км/с.

            Required data:
                Скорость_продольной_волны(Vp)

            Args:
                Скорость_продольной_волны(Vp) (np.ndarray[int/float]): массив со скоростью продольной волны(Vp) для проверки

            Returns:
                dict: Словарь, specification cловарь где ,result_mask - маска с результатом ,test_name - название теста ,
                      param_name - название параметра ,error_decr -краткое описание ошибки,well_name- название скважины,
                      MD - массив с глубинами
        """
        return self.__check_vp_vs(self.consts.vp, "test_vp", get_report, filters)

    def test_vs(self, get_report=True, filters=None) -> dict:
        """
            Тест предназначен для проверки физичности данных.
            В данном тесте проверяется соответствие интервалу 0.3<Vp<10 км/с.

            Required data:
                Скорость_поперечной_волны(Vs)

            Args:
                Скорость_поперечной_волны(Vs) (np.ndarray[int/float]): массив со скоростью поперечной волны(Vs) для проверки

            Returns:
                dict: Словарь, specification cловарь где ,result_mask - маска с результатом ,test_name - название теста ,
                      param_name - название параметра ,error_decr -краткое описание ошибки,well_name- название скважины,
                      MD - массив с глубинами
        """
        return self.__check_vp_vs(self.consts.vs, "test_vs", get_report, filters)

    def test_porosity_open(self, get_report=True, filters=None) -> dict:
        """
        Тест предназначен для проверки физичности данных.
        В данном тесте проверяется соответствие интервалу (0 ; 47,6]

        Required data:
            Кп_откр

        Args:
            Кп_откр (np.ndarray[int/float]): массив с кп_откр для проверки из переданной таблицы

        Returns:
            dict: Словарь, specification cловарь где ,result_mask - маска с результатом ,test_name - название теста ,
                      param_name - название параметра ,error_decr -краткое описание ошибки,well_name- название скважины,
                      MD - массив с глубинами
        """

        return self.__main_porosity_test(self.consts.kp_open, "test_porosity_open", get_report, filters)

    def test_porosity_abs(self, get_report=True, filters=None) -> dict:
        """
        Тест предназначен для проверки физичности данных.
        В данном тесте проверяется соответствие интервалу (0 ; 47,6]

        Required data:
            Кп_абс

        Args:
            Кп_абс (np.ndarray[int/float]): массив с кп_дин для проверки из переданной таблицы

        Returns:
            dict: Словарь, specification cловарь где ,result_mask - маска с результатом ,test_name - название теста ,
                      param_name - название параметра ,error_decr -краткое описание ошибки,well_name- название скважины,
                      MD - массив с глубинами
        """

        return self.__main_porosity_test(self.consts.kp_abs, "test_porosity_open", get_report, filters)

    def test_porosity_din(self, get_report=True, filters=None) -> dict:
        """
        Тест предназначен для проверки физичности данных.
        В данном тесте проверяется соответствие интервалу (0 ; 47,6]

        Required data:
            Кп_дин

        Args:
            Кп_дин (np.ndarray[int/float]): массив с кп_дин для проверки из переданной таблицы

        Returns:
            dict: Словарь, specification cловарь где ,result_mask - маска с результатом ,test_name - название теста ,
                      param_name - название параметра ,error_decr -краткое описание ошибки,well_name- название скважины,
                      MD - массив с глубинами
        """

        return self.__main_porosity_test(self.consts.kp_din, "test_porosity_din", get_report, filters)

    def test_porosity_eff(self, get_report=True, filters=None) -> dict:
        """
        Тест предназначен для проверки физичности данных.
        В данном тесте проверяется соответствие интервалу (0 ; 47,6]

        Required data:
            Кп_эфф

        Args:
            Кп_эфф (np.ndarray[int/float]): массив с кп_эфф для проверки из переданной таблицы

        Returns:
            dict: Словарь, specification cловарь где ,result_mask - маска с результатом ,test_name - название теста ,
                      param_name - название параметра ,error_decr -краткое описание ошибки,well_name- название скважины,
                      MD - массив с глубинами
        """

        return self.__main_porosity_test(self.consts.kp_eff, "test_porosity_eff", get_report, filters)

    def test_sw_residual(self, get_report=True, filters=None) -> dict:
        """
        Тест предназначен для проверки физичности данных.
        В данном тесте проверяется соответствие интервалу (0 ; 1]

        Required data:
            Кво

        Args:
            Кво (np.ndarray[int/float]): массив с коэффициентом остаточной водонасыщенности
                                        для проверки из переданной таблицы

        Returns:
            dict: Словарь, specification cловарь где ,result_mask - маска с результатом ,test_name - название теста ,
                      param_name - название параметра ,error_decr -краткое описание ошибки,well_name- название скважины,
                      MD - массив с глубинами
        """

        return self.__main_porosity_test(self.consts.kvo, "test_sw_residual", get_report, filters)

    def test_sowcr(self, get_report=True, filters=None) -> dict:
        """
        Тест предназначен для проверки физичности данных.
        В данном тесте проверяется соответствие интервалу (0 ; 1]

        Required data:
            Кно
        Args:
            self.sowcr (array[int/float]): массив с коэффициентом нефтенасыщенности для проверки
        Returns:
            dict: Словарь, specification cловарь где ,result_mask - маска с результатом ,test_name - название теста ,
                      param_name - название параметра ,error_decr -краткое описание ошибки
        """

        return self.__main_porosity_test(self.consts.kno, "test_sowcr", get_report, filters)

    def test_sg(self, get_report=True, filters=None) -> dict:
        """
        Тест предназначен для проверки физичности данных.
        В данном тесте проверяется соответствие интервалу (0 ; 1]

        Required data:
            Sg

        Args:
            Sg (np.ndarray[int/float]): массив с остаточной водонасыщенностью
                                        для проверки из переданной таблицы

        Returns:
            dict: Словарь, specification cловарь где ,result_mask - маска с результатом ,test_name - название теста ,
                      param_name - название параметра ,error_decr -краткое описание ошибки,well_name- название скважины,
                      MD - массив с глубинами
        """

        return self.__main_porosity_test(self.consts.sg, "test_sg", get_report, filters)

    def test_sgl(self, get_report=True, filters=None) -> dict:
        """
        Тест предназначен для проверки физичности данных.
        В данном тесте проверяется соответствие интервалу (0 ; 1]

        Required data:
            Sgl

        Args:
            Sgl (np.ndarray[int/float]): массив с связанной газонасыщенностью
                                            для проверки из переданной таблицы

        Returns:
            dict: Словарь, specification cловарь где ,result_mask - маска с результатом ,test_name - название теста ,
                      param_name - название параметра ,error_decr -краткое описание ошибки,well_name- название скважины,
                      MD - массив с глубинами
        """

        return self.__main_porosity_test(self.consts.sgl, "test_sgl", get_report, filters)

    def test_so(self, get_report=True, filters=None) -> dict:
        """
        Тест предназначен для проверки физичности данных.
        В данном тесте проверяется соответствие интервалу (0 ; 1]

        Required data:
            So

        Args:
            So (np.ndarray[int/float]): массив с остаточной водонасыщенностью для проверки

        Returns:
           dict: Словарь, specification cловарь где ,result_mask - маска с результатом ,test_name - название теста ,
                      param_name - название параметра ,error_decr -краткое описание ошибки,well_name- название скважины,
                      MD - массив с глубинами
        """

        return self.__main_porosity_test(self.consts.so, "test_so", get_report, filters)

    def test_sogcr(self, get_report=True, filters=None) -> dict:
        """
        Тест предназначен для проверки физичности данных.
        В данном тесте проверяется соответствие интервалу (0 ; 1]

        Required data:
            Sogcr

        Args:
            Sogcr (np.ndarray[int/float]): массив с критической нефтенасыщенностью
                                        для проверки из переданной таблицы

        Returns:
            dict: Словарь, specification cловарь где ,result_mask - маска с результатом ,test_name - название теста ,
                      param_name - название параметра ,error_decr -краткое описание ошибки,well_name- название скважины,
                      MD - массив с глубинами
        """

        return self.__main_porosity_test(self.consts.sogcr, "test_sogcr", get_report, filters)

    def test_sw(self, get_report=True, filters=None):
        """
        Тест предназначен для проверки физичности данных.
        В данном тесте проверяется соответствие интервалу (0 ; 1]

        Required data:
            Sw

        Args:
            Sw (np.ndarray[int/float]): массив с водонасыщенностью
                                            для проверки из переданной таблицы

        Returns:
            dict: Словарь, specification cловарь где ,result_mask - маска с результатом ,test_name - название теста ,
                      param_name - название параметра ,error_decr -краткое описание ошибки,well_name- название скважины,
                      MD - массив с глубинами
        """

        return self.__main_porosity_test(self.consts.sw, "test_sw", get_report, filters)

    def test_kpr_abs(self, get_report=True, filters=None) -> dict:
        """
        Тест предназначен для проверки физичности данных. Значение должно быть больше 0.

        Required data:
            Кпр_абс

        Args:
            Кпр_абс (np.ndarray[int/float]): массив с абсолютной проницаемостью для проверки из переданной таблицы

        Returns:
            dict: Словарь, specification cловарь где ,result_mask - маска с результатом ,test_name - название теста ,
                      param_name - название параметра ,error_decr -краткое описание ошибки,well_name- название скважины,
                      MD - массив с глубинами
        """

        return self.__main_greater_than_zero(self.consts.kpr_abs, "test_kpr_abs", get_report, filters)

    def test_kpr_abs_Y(self, get_report=True, filters=None) -> dict:
        """
        Тест предназначен для проверки физичности данных. Значение должно быть больше 0.

        Required data:
            Кпр_абс_Y

        Args:
            Кпр_абс_Y (np.ndarray[int/float]): массив с абсолютной проницаемостью по Y для проверки из переданной таблицы

        Returns:
            dict: Словарь, specification cловарь где ,result_mask - маска с результатом ,test_name - название теста ,
                      param_name - название параметра ,error_decr -краткое описание ошибки,well_name- название скважины,
                      MD - массив с глубинами
        """

        return self.__main_greater_than_zero(self.consts.kpr_abs_Y, "test_kpr_abs_Y", get_report, filters)

    def test_kpr_abs_Z(self, get_report=True, filters=None) -> dict:
        """
        Тест предназначен для проверки физичности данных. Значение должно быть больше 0.

        Required data:
            Кпр_абс_Z

        Args:
            Кпр_абс_Z (np.ndarray[int/float]): массив с абсолютной проницаемостью по Z для проверки из переданной таблицы

        Returns:
            dict: Словарь, specification cловарь где ,result_mask - маска с результатом ,test_name - название теста ,
                      param_name - название параметра ,error_decr -краткое описание ошибки,well_name- название скважины,
                      MD - массив с глубинами
        """

        return self.__main_greater_than_zero(self.consts.kpr_abs_Z, "test_kpr_abs_Z", get_report, filters)

    def test_kpr_rel(self, get_report=True, filters=None) -> dict:
        """
        Тест предназначен для проверки физичности данных. Значение должно быть больше 0.

        Required data:
            Кпр_отн

        Args:
            Кпр_отн (np.ndarray[int/float]): массив с относительной проницаемостью для проверки из переданной таблицы

        Returns:
            dict: Словарь, specification cловарь где ,result_mask - маска с результатом ,test_name - название теста ,
                      param_name - название параметра ,error_decr -краткое описание ошибки,well_name- название скважины,
                      MD - массив с глубинами
        """

        return self.__main_greater_than_zero(self.consts.kpr_rel, "test_kpr_rel", get_report, filters)

    def test_kpr_phase(self, get_report=True, filters=None) -> dict:
        """
        Тест предназначен для проверки физичности данных. Значение должно быть больше 0.

        Required data:
            Кпр_фазовая

        Args:
            Кпр_фазовая (np.ndarray[int/float]): массив с фазовой проницаемостью для проверки из переданной таблицы

        Returns:
            dict: Словарь, specification cловарь где ,result_mask - маска с результатом ,test_name - название теста ,
                      param_name - название параметра ,error_decr -краткое описание ошибки,well_name- название скважины,
                      MD - массив с глубинами
        """

        return self.__main_greater_than_zero(self.consts.kpr_phase, "test_kpr_phase", get_report, filters)

    def test_kpr_eff(self, get_report=True, filters=None) -> dict:
        """
        Тест предназначен для проверки физичности данных. Значение должно быть больше 0.

        Required data:
            Кпр_эфф

        Args:
            Кпр_эфф (np.ndarray[int/float]): массив с эффективной проницаемостью для проверки из переданной таблицы

        Returns:
            dict: Словарь, specification cловарь где ,result_mask - маска с результатом ,test_name - название теста ,
                      param_name - название параметра ,error_decr -краткое описание ошибки,well_name- название скважины,
                      MD - массив с глубинами
        """

        return self.__main_greater_than_zero(self.consts.kpr_eff, "test_kpr_eff", get_report, filters)

    def __main_poro_vs_param(self, poro_name, param_name, filters, test_name, get_report):
        clear_df, index, well_name, md = self.__get_data_from_data_kern(param=[poro_name, param_name], filters=filters)
        porosity = np.array(clear_df[poro_name])
        param = np.array(clear_df[param_name])
        check_result_for_first_param, wrong_for_first_param, check_text_for_first_param = self.__check_data(
            porosity, get_report)
        check_result_for_second_param, wrong_for_second_param, check_text_for_second_param = self.__check_data(
            param, get_report)

        if check_result_for_first_param and check_result_for_second_param:
            r2 = self.test_general_dependency_checking(porosity, param)["specification"]["r2"]
            result = True
            a, b = linear_dependence_function(porosity, param)
            if a >= 0 or r2 < 0.7:
                result = False

            wrong_values = dropdown_search(porosity,
                                           param,
                                           a, b)
            linear_function_visualization(porosity,
                                          param,
                                          a, b,
                                          r2,
                                          get_report,
                                          poro_name,
                                          param_name,
                                          test_name,
                                          wrong_values)
            wrong_values = np.where(wrong_values, 1, result)

            text = self.consts.dependency_accepted + str(wrong_values) if result \
                else self.consts.dependency_wrong + str(wrong_values)

            self.__generate_report(text, result, get_report)
            self.data_kern.mark_errors(poro_name, test_name, text, wrong_values, index)

            return self.__generate_returns_dict(check_result_for_first_param and check_text_for_second_param, result,
                                                wrong_values, text, well_name, md,
                                                test_name, [poro_name, param_name], r2=r2)
        else:
            self.data_kern.mark_errors(param_name, test_name,
                                       check_text_for_first_param + " " + check_text_for_second_param,
                                       wrong_for_first_param if len(wrong_for_first_param) != 0
                                       else wrong_for_second_param, index)
            return self.__generate_returns_dict(check_result_for_first_param and check_text_for_second_param, False,
                                                wrong_for_first_param if len(wrong_for_first_param) != 0
                                                else wrong_for_second_param,
                                                check_text_for_first_param + " " + check_text_for_second_param,
                                                well_name,
                                                md,
                                                test_name, [poro_name, param_name])

    def test_general_dependency_checking(self, x, y, get_report=True):
        """
            Тест предназначен для оценки дисперсии входных данных.
            Он проводится по следующему алгоритму: изначально,
            используя статистические методы, детектируются и удаляются
            выбросные точки, затем полученное облако точек аппроксимируется
            и считается коэффициент детерминации R2. Если его значение больше
            0.7, то тест считается пройденным. Если значение меньше 0.7, то
            точки сортируются по удаленности от линии тренда, и запускается
            цикл, за одну итерацию которого удаляется самая отдаленная от
            линии аппроксимации точка, и считается R2, если значение больше
            0.7, и удалено менее 10% точек, то тест пройден, иначе - нет.

            Args:
                x (np.ndarray[int/float]): массив с данными для проверки
                y (np.ndarray[int/float]): массив с данными для проверки

            Returns:
                dic: Словарь с результатом теста, коэффициентом r2
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

    def test_porosity_open_vs_swl(self, get_report=True, filters=None):
        """
        Тест предназначен для оценки соответствия типовой
        для данного кроссплота и полученной аппроксимации.
        В данном случае зависимость линейная по функции
        y=a*x+b, при этом a<0

        Required data:
            Кво; Кп_откр

        Args:
            Кво (np.ndarray[int/float]): массив с данными коэффициент остаточной водонасыщенности для проверки
            Кп_откр (np.ndarray[int/float]): массив с данными Кп откр для проверки

        Returns:
            Словарь, specification cловарь где ,result_mask - маска с результатом ,test_name - название теста ,
                      param_name - названия параметров ,error_decr -краткое описание ошибки,well_name- название скважины,
                      MD - массив с глубинами
        """
        return self.__main_poro_vs_param(self.consts.kp_open, self.consts.kvo, filters, "test_porosity_open_vs_swl",
                                         get_report)

    def test_kp_abs_vs_swl(self, get_report=True, filters=None):
        """
        Тест предназначен для оценки соответствия типовой
        для данного кроссплота и полученной аппроксимации.
        В данном случае зависимость линейная по функции
        y=a*x+b, при этом a<0

        Required data:
            Кво; Кп_абс

        Args:
            Кво (np.ndarray[int/float]): массив с данными коэффициент остаточной водонасыщенности для проверки
            Кп_абс (np.ndarray[int/float]): массив с данными Кп абс для проверки
                Returns:
            Словарь, specification cловарь где ,result_mask - маска с результатом ,test_name - название теста ,
                      param_name - названия параметров ,error_decr -краткое описание ошибки,well_name- название скважины,
                      MD - массив с глубинами
        """

        return self.__main_poro_vs_param(self.consts.kp_abs, self.consts.kvo, filters, "test_kp_abs_vs_swl",
                                         get_report)

    def test_kp_open_vs_density_dry(self, get_report=True, filters=None):
        """
        Тест предназначен для оценки соответствия типовой
        для данного кроссплота и полученной аппроксимации.
        В данном случае зависимость линейная по функции
        y=a*x+b, при этом a<0

        Required data:
            Кп_откр; Плотность_абсолютно_сухого_образца

        Args:
            Кп_откр (np.ndarray[int/float]): массив с данными открытая пористость для проверки
            Плотность_абсолютно_сухого_образца (np.ndarray[int/float]): массив с данными
                                                                        плотность_абсолютно_сухого_образца для проверки

        Returns:
            Словарь, specification словарь где, result_mask - маска с результатом, test_name - название теста,
                      param_name - названия параметров, error_decr -краткое описание ошибки, well_name- название скважины,
                      MD - массив с глубинами
        """
        return self.__main_poro_vs_param(self.consts.kp_open, self.consts.ads_density, filters,
                                         "test_kp_open_vs_density_dry", get_report)

    def test_kp_open_vs_density_wet(self, get_report=True, filters=None):
        """
        Тест предназначен для оценки соответствия типовой
        для данного кроссплота и полученной аппроксимации.
        В данном случае зависимость линейная по функции
        y=a*x+b, при этом a<0

        Required data:
            Кп_откр; Плотность_максимально_увлажненного_образца

        Args:
            Кп_откр (np.ndarray[int/float]): массив с данными открытая пористость для проверки
            Плотность_максимально_увлажненного_образца(np.ndarray[int/float]): массив с данными
                                                                                плотность_максимально_увлажненного_образца для проверки

        Returns:
            Словарь, specification словарь где, result_mask - маска с результатом, test_name - название теста,
                      param_name - названия параметров, error_decr -краткое описание ошибки, well_name- название скважины,
                      MD - массив с глубинами
        """
        return self.__main_poro_vs_param(self.consts.kp_open, self.consts.mms_density, filters,
                                         "test_kp_open_vs_density_wet", get_report)

    def test_kp_abs_vs_density_dry(self, get_report=True, filters=None):
        """
        Тест предназначен для оценки соответствия типовой
        для данного кроссплота и полученной аппроксимации.
        В данном случае зависимость линейная по функции
        y=a*x+b, при этом a<0

        Required data:
            Кп_абс; Плотность_абсолютно_сухого_образца

        Args:
            Кп_абс (np.ndarray[int/float]): массив с данными коэффициент остаточной водонасыщенности для проверки
            Плотность_абсолютно_сухого_образца (np.ndarray[int/float]): массив с данными
                                                                        плотность_абсолютно_сухого_образца для проверки

        Returns:
            Словарь, specification словарь где, result_mask - маска с результатом, test_name - название теста,
                      param_name - названия параметров, error_decr -краткое описание ошибки, well_name- название скважины,
                      MD - массив с глубинами
        """
        return self.__main_poro_vs_param(self.consts.kp_abs, self.consts.ads_density, filters,
                                         "test_kp_abs_vs_density_dry", get_report)

    def test_kp_abs_vs_density_wet(self, get_report=True, filters=None):
        """
        Тест предназначен для оценки соответствия типовой
        для данного кроссплота и полученной аппроксимации.
        В данном случае зависимость линейная по функции
        y=a*x+b, при этом a<0

        Required data:
            Кп_абс; Плотность_максимально_увлажненного_образца

        Args:
            Кп_абс (np.ndarray[int/float]): массив с данными абсолютной пористости для проверки
            Плотность_максимально_увлажненного_образца(np.ndarray[int/float]): массив с данными
                                                                                плотность_максимально_увлажненного_образца для проверки

        Returns:
            Словарь, specification словарь где, result_mask - маска с результатом, test_name - название теста,
                      param_name - названия параметров, error_decr -краткое описание ошибки, well_name- название скважины,
                      MD - массив с глубинами
        """
        return self.__main_poro_vs_param(self.consts.kp_abs, self.consts.mms_density, filters,
                                         "test_kp_abs_vs_density_wet", get_report)

    def __generate_dependency_result(self, first_param: np.array, second_param: np.array, a: int, b: int, r2: int,
                                     get_report: bool, first_name: str,
                                     second_name: str, test_name: str, result: bool) -> np.array:
        """

        Args:
            first_param(np.ndarray[int/float]): значения по оси X
            second_param(np.ndarray[int/float]): значения по оси Y
            a(int): коэффициент наклона
            b(int): коэффициент сдвига
            r2(int): коэффициент r2
            get_report(bool): флаг для отображения отчета
            first_name(string): название параметра по оси X
            second_name(string): название параметра по оси Y
            test_name(string): название теста
            result(bool): результат теста

        Returns:
            wrong_values(np.ndarray[int/float]):маска с выпадающими значениями
        """

        wrong_values = dropdown_search(first_param,
                                       second_param,
                                       a, b)
        linear_function_visualization(first_param,
                                      second_param,
                                      a, b,
                                      r2,
                                      get_report,
                                      first_name,
                                      second_name,
                                      test_name,
                                      wrong_values)
        wrong_values = np.where(wrong_values, 1, result)
        return wrong_values

    def __main_poro_vs_density(self, mineral_porosity: str, volume_porosity: str, test_name: str, get_report=True,
                               filters=None):
        """
        Тест предназначен для проверки связи между двумя кросс плотами - Обплнас-Кп и Минпл-Кп.
        Данная взаимосвязь описывается линией тренда : y=a1*x+b1, при этом a1<a2,
        где а2 - коэффициент из зависимости Минпл-Кп

        Args:
            mineral_porosity (string): название пористости для зависимости Минпл-Кп
            volume_porosity(string): название пористости для зависимости Обплнас-Кп
            test_name (string): название теста

        Returns:
            Словарь, specification словарь где, result_mask - маска с результатом, test_name - название теста,
                      param_name - названия параметров, error_decr -краткое описание ошибки, well_name- название скважины,
                      MD - массив с глубинами
        """
        clear_df_kp_mineral, index, well_name, md = self.__get_data_from_data_kern(param=[mineral_porosity,
                                                                                          self.consts.mineral_density],
                                                                                   filters=filters)
        clear_df_kp_volume, index, well_name, md = self.__get_data_from_data_kern(param=[volume_porosity,
                                                                                         self.consts.volume_density],
                                                                                  filters=filters)

        poro_mineral = np.array(clear_df_kp_mineral[mineral_porosity])
        poro_volume = np.array(clear_df_kp_volume[volume_porosity])
        volume_density = np.array(clear_df_kp_volume[self.consts.volume_density])
        mineral_density = np.array(clear_df_kp_mineral[self.consts.mineral_density])
        check_result_for_poro_mineral, wrong_for_poro_mineral, check_text_for_poro_mineral = self.__check_data(
            poro_mineral, get_report)
        check_result_for_poro_volume, wrong_for_poro_volume, check_text_for_poro_volume = self.__check_data(
            poro_volume, get_report)
        check_result_for_volume_density, wrong_for_volume_density, check_text_for_volume_density = self.__check_data(
            volume_density, get_report)
        check_result_for_mineral_density, wrong_for_mineral_density, check_text_for_mineral_density = self.__check_data(
            mineral_density, get_report)
        check_result = check_result_for_poro_mineral and check_result_for_poro_volume and check_result_for_volume_density and check_result_for_mineral_density
        if check_result:
            a_poro_vs_volume, b_poro_vs_volume = linear_dependence_function(volume_density, poro_volume)
            a_poro_vs_mineral, b_poro_vs_mineral = linear_dependence_function(mineral_density, poro_mineral)
            r2_volume = self.test_general_dependency_checking(poro_volume, volume_density)["specification"]["r2"]
            r2_mineral = self.test_general_dependency_checking(poro_mineral, mineral_density)["specification"]["r2"]

            result = True

            if a_poro_vs_mineral >= a_poro_vs_volume or r2_volume < 0.7 or r2_mineral < 0.7:
                result = False

            wrong_values_volume = self.__generate_dependency_result(volume_density,
                                                                    poro_volume,
                                                                    a_poro_vs_volume,
                                                                    b_poro_vs_volume,
                                                                    r2_volume,
                                                                    get_report,
                                                                    self.consts.volume_density,
                                                                    volume_porosity,
                                                                    test_name,
                                                                    result)

            wrong_values_mineral = self.__generate_dependency_result(mineral_density,
                                                                     poro_mineral,
                                                                     a_poro_vs_mineral,
                                                                     b_poro_vs_mineral,
                                                                     r2_mineral,
                                                                     get_report,
                                                                     self.consts.mineral_density,
                                                                     mineral_porosity,
                                                                     test_name,
                                                                     result)

            text = self.consts.dependency_accepted + str(wrong_values_volume) + " " + str(
                wrong_values_mineral) if result \
                else self.consts.dependency_wrong + str(wrong_values_volume) + " " + str(wrong_values_mineral)
            self.__generate_report(text, result, get_report)
            self.data_kern.mark_errors(volume_porosity, test_name, text, wrong_values_volume, index)
            self.__generate_report(text, result, get_report)
            self.data_kern.mark_errors(mineral_porosity, test_name, text, wrong_values_mineral, index)

            return self.__generate_returns_dict(check_result, result,
                                                wrong_values_volume + wrong_values_mineral,
                                                text, well_name, md,
                                                test_name, [poro_mineral, poro_volume,
                                                            self.consts.volume_density,
                                                            self.consts.mineral_density],
                                                r2=str(r2_volume) + " " + str(r2_mineral))
        else:
            if len(wrong_for_poro_mineral) != 0:
                check_text = check_text_for_poro_mineral
                param_name = mineral_porosity
                wrong_values = wrong_for_poro_mineral
            elif len(wrong_for_poro_volume) != 0:
                check_text = check_text_for_poro_volume
                param_name = volume_porosity
                wrong_values = wrong_for_poro_mineral
            elif len(wrong_for_volume_density) != 0:
                check_text = check_text_for_volume_density
                param_name = self.consts.volume_density
                wrong_values = wrong_for_volume_density
            else:
                check_text = check_text_for_mineral_density
                param_name = self.consts.mineral_density
                wrong_values = wrong_for_mineral_density

            self.data_kern.mark_errors(param_name, test_name, check_text,
                                       wrong_values, index)
            return self.__generate_returns_dict(check_result, False,
                                                wrong_values,
                                                check_text, well_name, md,
                                                test_name, [poro_mineral, poro_volume,
                                                            self.consts.volume_density,
                                                            self.consts.mineral_density])

    def test_poro_abs_vs_density(self, get_report=True, filters=None):
        """
        Тест предназначен для проверки физичности взаимосвязи
        двух кросс-плотов - Обплнас-Кп и Минпл-Кп.
        Пусть первый аппроксимируется линией тренда y=a1*x+b1,
        а второй - y=a2*x+b2, при этом a1<a2

        Required data:
            Кп_абс; Минералогическая плотность, г/см3; Объемная плотность, г/см3

        Args:
            Кп_абс (np.ndarray[int/float]): массив с данными абсолютной пористости для проверки
            Минералогическая плотность, г/см3(np.ndarray[int/float]): массив с данными
                                                                                миниралогической плотности для проверки
            Объемная плотность, г/см3(np.ndarray[int/float]): массив с данными объемной плотности для проверки

        Returns:
            Словарь, specification словарь где, result_mask - маска с результатом, test_name - название теста,
                      param_name - названия параметров, error_decr -краткое описание ошибки, well_name- название скважины,
                      MD - массив с глубинами
        """
        return self.__main_poro_vs_density(self.consts.kp_abs, self.consts.kp_abs, "test_poro_abs_open_vs_density",
                                           get_report, filters)

    def test_poro_open_vs_density(self, get_report=True, filters=None):
        """
        Тест предназначен для проверки физичности взаимосвязи
        двух кросс-плотов - Обплнас-Кп и Минпл-Кп.
        Пусть первый аппроксимируется линией тренда y=a1*x+b1,
        а второй - y=a2*x+b2, при этом a1<a2

        Required data:
            Кп_откр; Минералогическая плотность, г/см3; Объемная плотность, г/см3

        Args:
            Кп_откр (np.ndarray[int/float]): массив с данными открытой пористости для проверки
            Минералогическая плотность, г/см3(np.ndarray[int/float]): массив с данными
                                                                                миниралогической плотности для проверки
            Объемная плотность, г/см3(np.ndarray[int/float]): массив с данными объемной плотности для проверки

        Returns:
            Словарь, specification словарь где, result_mask - маска с результатом, test_name - название теста,
                      param_name - названия параметров, error_decr -краткое описание ошибки, well_name- название скважины,
                      MD - массив с глубинами
        """
        return self.__main_poro_vs_density(self.consts.kp_open, self.consts.kp_open, "test_poro_open_vs_density",
                                           get_report, filters)

    def test_poro_abs_mineral_vs_poro_open_volume(self, get_report=True, filters=None):
        """
        Тест предназначен для проверки физичности взаимосвязи
        двух кросс-плотов - Обплнас-Кп и Минпл-Кп.
        Пусть первый аппроксимируется линией тренда y=a1*x+b1,
        а второй - y=a2*x+b2, при этом a1<a2

        Required data:
            Кп_откр; Кп_абс; Минералогическая плотность, г/см3; Объемная плотность, г/см3

        Args:
            Кп_откр (np.ndarray[int/float]): массив с данными открытой пористости для связи Минпл-Кп
            Кп_абс (np.ndarray[int/float]): массив с данными абсолютной пористости для связи Обплнас-Кп
            Минералогическая плотность, г/см3(np.ndarray[int/float]): массив с данными
                                                                                миниралогической плотности для проверки
            Объемная плотность, г/см3(np.ndarray[int/float]): массив с данными объемной плотности для проверки

        Returns:
            Словарь, specification словарь где, result_mask - маска с результатом, test_name - название теста,
                      param_name - названия параметров, error_decr -краткое описание ошибки, well_name- название скважины,
                      MD - массив с глубинами
        """
        return self.__main_poro_vs_density(self.consts.kp_abs, self.consts.kp_open,
                                           "test_poro_abs_mineral_vs_poro_open_volume", get_report, filters)

    def test_poro_open_mineral_vs_poro_abs_volume(self, get_report=True, filters=None):
        """
        Тест предназначен для проверки физичности взаимосвязи
        двух кросс-плотов - Обплнас-Кп и Минпл-Кп.
        Пусть первый аппроксимируется линией тренда y=a1*x+b1,
        а второй - y=a2*x+b2, при этом a1<a2

        Required data:
            Кп_откр; Кп_абс; Минералогическая плотность, г/см3; Объемная плотность, г/см3

        Args:
            Кп_откр (np.ndarray[int/float]): массив с данными открытой пористости для связи Обплнас-Кп
            Кп_абс (np.ndarray[int/float]): массив с данными абсолютной пористости для связи Минпл-Кп
            Минералогическая плотность, г/см3(np.ndarray[int/float]): массив с данными
                                                                                миниралогической плотности для проверки
            Объемная плотность, г/см3(np.ndarray[int/float]): массив с данными объемной плотности для проверки

        Returns:
            Словарь, specification словарь где, result_mask - маска с результатом, test_name - название теста,
                      param_name - названия параметров, error_decr -краткое описание ошибки, well_name- название скважины,
                      MD - массив с глубинами
        """
        return self.__main_poro_vs_density(self.consts.kp_open, self.consts.kp_abs,
                                           "test_poro_open_mineral_vs_poro_abs_volume", get_report, filters)

    def __main_poro_vs_perm_abs(self, poro_name, filters, test_name, get_report):
        """
        Тест применяется для сравнения двух аппроксимаций:
        характерной (эталонной для выбранного набора данных)
        и текущей.  Характерной зависимостью является
        линейная по функции y=a*exp(b*x) при этом b>0

        Args:
            poro_name (string): название пористости
            test_name(string): название теста
            filters(array[dic]): применяемые фильтры в формате [{"name":str,"value":str||int,
                                                                "operation"(np.ndarray[string]):[=, !=, >, <, >=, <=]}]

        Returns:
            Словарь, specification словарь где, result_mask - маска с результатом, test_name - название теста,
            param_name - названия параметров, error_decr -краткое описание ошибки, well_name- название скважины,
            MD - массив с глубинами
        """
        clear_df, index, well_name, md = self.__get_data_from_data_kern(param=[poro_name, self.consts.kpr_abs],
                                                                        filters=filters)
        porosity = np.array(clear_df[poro_name])
        perm = np.array(clear_df[self.consts.kpr_abs])
        check_result_for_first_param, wrong_for_first_param, check_text_for_first_param = self.__check_data(
            porosity, get_report)
        check_result_for_second_param, wrong_for_second_param, check_text_for_second_param = self.__check_data(
            perm, get_report)

        if check_result_for_first_param and check_result_for_second_param:
            r2 = self.test_general_dependency_checking(porosity, np.log(perm))["specification"]["r2"]
            result = True
            a, b = exponential_function(porosity, perm)
            if b <= 0 or r2 < 0.7:
                result = False

            wrong_values = dropdown_search(porosity,
                                           np.log(perm),
                                           a, b)
            linear_function_visualization(porosity,
                                          np.log(perm),
                                          a, b,
                                          r2,
                                          get_report,
                                          poro_name,
                                          self.consts.kpr_abs,
                                          test_name,
                                          wrong_values)
            wrong_values = np.where(wrong_values, 1, result)

            text = self.consts.dependency_accepted + str(wrong_values) if result \
                else self.consts.dependency_wrong + str(wrong_values)

            self.__generate_report(text, result, get_report)
            self.data_kern.mark_errors(poro_name, test_name, text, wrong_values, index)

            return self.__generate_returns_dict(check_result_for_first_param and check_text_for_second_param, result,
                                                wrong_values, text, well_name, md,
                                                test_name, [poro_name, perm], r2=r2)
        else:
            self.data_kern.mark_errors(perm, test_name,
                                       check_text_for_first_param + " " + check_text_for_second_param,
                                       wrong_for_first_param if len(wrong_for_first_param) != 0
                                       else wrong_for_second_param, index)
            return self.__generate_returns_dict(check_result_for_first_param and check_text_for_second_param, False,
                                                wrong_for_first_param if len(wrong_for_first_param) != 0
                                                else wrong_for_second_param,
                                                check_text_for_first_param + " " + check_text_for_second_param,
                                                well_name,
                                                md,
                                                test_name, [poro_name, perm])

    def test_kpr_abs_vs_kp_open(self, get_report=True, filters=None):
        """
        Тест применяется для сравнения двух аппроксимаций:
        характерной (эталонной для выбранного набора данных)
        и текущей.  Характерной зависимостью является
        линейная по функции y=a*exp(b*x) при этом b>0

        Required data:
            Кпр_абс; Кп_откр

        Args:
            Кпр_абс (np.ndarray[int/float]): массив с данными абсолютной проницаемости для проверки
            Кп_откр (np.ndarray[int/float]): массив с данными открытой пористости для проверки

        Returns:
            Словарь, specification словарь где, result_mask - маска с результатом, test_name - название теста,
            param_name - названия параметров, error_decr -краткое описание ошибки, well_name- название скважины,
            MD - массив с глубинами
        """
        return self.__main_poro_vs_perm_abs(self.consts.kp_open, filters, "perm_abs_vs_kp_open", get_report)

    def test_kpr_abs_vs_kp_abs(self, get_report=True, filters=None):
        """
        Тест применяется для сравнения двух аппроксимаций:
        характерной (эталонной для выбранного набора данных)
        и текущей.  Характерной зависимостью является
        линейная по функции y=a*exp(b*x) при этом b>0

        Required data:
            Кпр_абс; Кп_абс

        Args:
            Кпр_абс (np.ndarray[int/float]): массив с данными абсолютной проницаемости для проверки
            Кп_абс(np.ndarray[int/float]): массив с данными абсолютной пористости для проверки

        Returns:
            Словарь, specification словарь где, result_mask - маска с результатом, test_name - название теста,
            param_name - названия параметров, error_decr -краткое описание ошибки, well_name- название скважины,
            MD - массив с глубинами
        """
        return self.__main_poro_vs_perm_abs(self.consts.kp_abs, filters, "test_kpr_abs_vs_kp_abs", get_report)

    def __find_duplicate_indices(self, arr) -> list[tuple]:
        """
        Находит индексы попарно одинаковых значений в массиве

        Args:
            arr (np.ndarray[int/float]): входной массив

        Returns:
            array[tuple]: список кортежей с индексами попарно одинаковых значений
        """
        # Создаем словарь для хранения индексов
        index_dict = {value: [] for value in arr}

        # Заполняем словарь индексами
        for i, value in enumerate(arr):
            index_dict[value].append(i)

        # Формируем список кортежей с индексами попарно одинаковых значений
        duplicate_indices = [(idx1, idx2) for indices in index_dict.values() for idx1, idx2 in
                             zip(indices, indices[1:])]

        return duplicate_indices

    def __find_difference_in_duplicate_indices(self, duplicate_indices1, duplicate_indices2) -> list[tuple]:

        """
        Находит различия между двумя списками duplicate_indices и возвращает отличающиеся пары

        Args:
            duplicate_indices1 (array[tuple]): первый список кортежей с индексами
            duplicate_indices2 (array[tuple]): второй список кортежей с индексами

        Returns:
             array[tuple]: отличающиеся пары индексов
        """
        set1 = set(duplicate_indices1)
        set2 = set(duplicate_indices2)

        difference = (set1 | set2) - (set1 & set2)
        return list(difference)

    def __create_mask(self, array, diff) -> np.ndarray:

        """
        Создает маску, где 1 помечены пары значений, которые не совпадают

        Args:
            array (np.ndarray[int/float]): массив для пометки совпадений
            diff (array[tuple]): список кортежей с индексами повторяющихся значений

        Returns:
            mask(np.ndarray[int]): маска для array

        """

        mask = np.zeros_like(array)

        for indices in diff:
            mask[indices[0]] = 1
            mask[indices[1]] = 1
        return mask

    def __main_data_tampering(self, first_param_name: str, second_param_name: str,
                              filters) -> tuple[bool, np.ndarray, np.ndarray, str, np.ndarray]:

        """
        Основной тест для проверки подлога данных

        Args:
            first_param_name(string): название первого параметра
            second_param_name(string): название второго параметра

        Returns:
            result(bool): результат проверки
            mask(np.ndarray(int): маска с результатом
            index(np.ndarray(int): индексы данных по глубине после удаления nan
            well_name(string): название скважины
            md(np.ndarray(int): массив с глубинами для которых взяты данные
        """

        clear_df, index, well_name, md = self.__get_data_from_data_kern(
            param=[first_param_name, second_param_name],
            filters=filters)
        first_param = np.array(clear_df[first_param_name])
        second_param = np.array(clear_df[second_param_name])
        duplicate_indices_first_param = self.__find_duplicate_indices(first_param)
        duplicate_indices_second_param = self.__find_duplicate_indices(second_param)
        diff = self.__find_difference_in_duplicate_indices(duplicate_indices_first_param,
                                                           duplicate_indices_second_param)
        mask = self.__create_mask(first_param, diff)
        result = np.sum(mask) == 0
        return result, mask, index, well_name, md

    def test_data_tampering(self, get_report=True, filters=None):
        """
        Required data:
            Кпр_абс; Кп_откр; Кво; Параметр_пористости(F); Параметр_насыщенности(RI); Плотность_абсолютно_сухого_образца;
            Плотность_максимально_увлажненного_образца

        Args:
            Кпр_абс(np.ndarray[int/float]): массив с данными абсолютной проницаемости для проверки
            Кп_откр(np.ndarray[int/float]): массив с данными открытой пористости для проверки
            Кво(np.ndarray[int/float]): массив с данными коэффициент остаточной водонасыщенности для проверки
            Параметр_пористости(F)(np.ndarray[int/float]): массив с данными параметра пористости для проверки
            Параметр_насыщенности(RI)(np.ndarray[int/float]): массив с данными параметра насыщенности для проверки
            Плотность_абсолютно_сухого_образца(np.ndarray[int/float]): массив с данными плотности абсолютно сухого образца
            Плотность_максимально_увлажненного_образца(np.ndarray[int/float]): массив с данными максимально увлажненного образца

        Returns:
            Словарь, specification словарь где, result_mask - словарь с результатом
            {"название параметра/пары параметров":маска с результатом}, test_name - название теста,
            param_name - названия параметров, error_decr -краткое описание ошибки, well_name- название скважины,
            MD - массив с глубинами
        """
        result_masks = {}
        final_result = True
        well_name, md = "", []

        for key, value in self.parameters_to_compare.items():
            result, mask, index, well_name, md = self.__main_data_tampering(key, value, filters)
            result_masks[key + " and " + value] = mask.tolist()
            final_result *= result
            text = self.consts.data_tampering_accepted if result else self.consts.data_tampering_wrong
            self.data_kern.mark_errors(key, "test_data_tampering", text, mask, index)
            self.data_kern.mark_errors(value, "test_data_tampering", text, mask, index)

        final_text = self.consts.data_tampering_accepted if final_result else self.consts.data_tampering_wrong

        self.__generate_report(final_text, final_result, get_report)
        return self.__generate_returns_dict(True, final_result,
                                            result_masks, final_text, well_name, md,
                                            "test_data_tampering", [list(self.parameters_to_compare.keys())])

    def __create_errors_dict(self, param_names, wrong_arrays, check_texts) -> dict:
        """
        Метод предназначен для создания словаря с ошибками
        Args:
            param_names(list[string]): массив с названиями проверяемых значений
            wrong_arrays(list[np.ndarray[int]]): массив, содержащий массив с ошибочными значениями
            check_texts(list[string]): массив с текстами ошибок

        Returns:
            dict({"string":tuple}) - словарь с ошибками, где ключ название параметра, а значение
             пара из массива с ошибками и описанием ошибки

        """
        errors_dict = {}

        for i, param_name in enumerate(param_names):
            errors_dict[param_name] = (wrong_arrays[i], check_texts[i])
        return errors_dict

    def __get_first_wrong_array(self, param_names, wrong_arrays, check_texts) -> tuple[str, np.ndarray, str]:
        """
        Метод предназначен для поиска первого провалившего проверку параметра
        Args:
            param_names(list[string]): массив с названиями проверяемых значений
            wrong_arrays(list[np.ndarray[int]]): массив, содержащий массив с ошибочными значениями
            check_texts(list[string]): массив с текстами ошибок

        Returns:
            key(string): название ошибочного параметра
            error_array(np.ndarray[int]): массив с ошибочными значениями
            error_description(string): описание ошибки

        """
        errors_dict = self.__create_errors_dict(param_names, wrong_arrays, check_texts)
        for key, (error_array, error_description) in errors_dict.items():
            if len(error_array) > 0:
                return key, error_array, error_description

    def __main_poro_perm_cut_off(self, poro_name, test_name, get_report=True, filters=None) -> dict:
        """
        Основной тест для проверки соответствия ФЕС и критериев отсечения
        Если Кп>Кп_cut_off то тогда Кпр>Кпр_cut_off

        Args:
            poro_name(string):название используемой пористости
            test_name(string): название теста

        Returns:
            Словарь, specification словарь где, result_mask - словарь с результатом
            {"название параметра/пары параметров":маска с результатом}, test_name - название теста,
            param_name - названия параметров, error_decr -краткое описание ошибки, well_name- название скважины,
            MD - массив с глубинами

        """
        clear_df, index, well_name, md = self.__get_data_from_data_kern(
            param=[poro_name, self.consts.kpr_abs],
            filters=filters)

        cut_off_df, _, _, _ = self.__get_data_from_data_kern(param=[self.consts.cut_off_poro, self.consts.cut_off_perm],
                                                             filters=None)
        result_dict = {}
        poro_param = np.array(clear_df[poro_name])
        kpr_param = np.array(clear_df[self.consts.kpr_abs])
        poro_cut_off = np.array(cut_off_df[self.consts.cut_off_poro])
        perm_cut_off = np.array(cut_off_df[self.consts.cut_off_perm])
        check_result_for_poro_param, wrong_for_poro_param, check_text_for_poro_param = self.__check_data(
            poro_param, get_report)
        check_result_for_kpr_param, wrong_for_kpr_param, check_text_for_kpr_param = self.__check_data(
            kpr_param, get_report)
        check_result_for_poro_cut_off, wrong_for_poro_cut_off, check_text_for_poro_cut_off = self.__check_data(
            poro_cut_off, get_report)
        check_result_for_perm_cut_off, wrong_for_perm_cut_off, check_text_for_perm_cut_off = self.__check_data(
            perm_cut_off, get_report)
        check_result = check_result_for_poro_param and check_result_for_kpr_param \
                       and check_result_for_poro_cut_off and check_result_for_perm_cut_off
        if check_result:
            indices_poro = np.where(poro_param > poro_cut_off)[0]
            mask = (kpr_param[indices_poro] < perm_cut_off)
            result_mask = np.zeros_like(poro_param)
            result_mask[indices_poro[mask]] = 1
            result = np.sum(result_mask) == 0
            text = self.consts.cut_off_accepted if result else self.consts.cut_off_wrong
            self.__generate_report(text, result, get_report)
            self.data_kern.mark_errors(poro_name, test_name, text, result_mask, index)
            self.data_kern.mark_errors(self.consts.kpr_abs, test_name, text, result_mask, index)
            result_dict[poro_name] = result_mask.tolist()
            result_dict[self.consts.kpr_abs] = result_mask.tolist()
            return self.__generate_returns_dict(check_result, result,
                                                result_dict, text, well_name, md,
                                                test_name, [poro_name, self.consts.kpr_abs, self.consts.cut_off_poro,
                                                            self.consts.cut_off_perm])

        else:
            param_name, result_mask, text = self.__get_first_wrong_array(
                [poro_name, self.consts.kpr_abs, self.consts.cut_off_poro, self.consts.cut_off_perm],
                [wrong_for_poro_param, wrong_for_kpr_param, wrong_for_poro_cut_off, wrong_for_perm_cut_off],
                [check_text_for_poro_param, check_text_for_kpr_param, check_text_for_poro_cut_off,
                 check_text_for_perm_cut_off])
            self.data_kern.mark_errors(param_name, test_name, text, result_mask, index)
            return self.__generate_returns_dict(check_result, False,
                                                result_mask,
                                                text,
                                                well_name,
                                                md,
                                                test_name, [poro_name, self.consts.kpr_abs, self.consts.cut_off_poro,
                                                            self.consts.cut_off_perm])

    def test_cut_off_kp_open(self, get_report=True, filters=None) -> dict:
        """
        Тест предназначен для проверки соответствия ФЕС и критериев отсечения
        Если Кп>Кп_cut_off то тогда Кпр>Кпр_cut_off

        Required data:
            Кпр_абс; Кп_откр; Cut-off_проницаемость; Cut-off_пористость

        Args:
            Кпр_абс(np.ndarray[int/float]): массив с данными абсолютной проницаемости для проверки
            Кп_откр(np.ndarray[int/float]): массив с данными открытой пористости для проверки
            Cut-off_проницаемость(np.ndarray[int/float]): значение критерия отсечения для проницаемости
            Cut-off_пористость(F)(np.ndarray[int/float]): значения критерия отсечения для пористости

        Returns:
            Словарь, specification словарь где, result_mask - словарь с результатом
            {"название параметра/пары параметров":маска с результатом}, test_name - название теста,
            param_name - названия параметров, error_decr -краткое описание ошибки, well_name- название скважины,
            MD - массив с глубинами
        """
        return self.__main_poro_perm_cut_off(self.consts.kp_open, "cut_off_kp_open", get_report, filters)

    def test_cut_off_kp_abs(self, get_report=True, filters=None) -> dict:
        """
        Тест предназначен для проверки соответствия ФЕС и критериев отсечения
        Если Кп>Кп_cut_off то тогда Кпр>Кпр_cut_off

        Required data:
            Кпр_абс; Кп_откр; Cut-off_проницаемость; Cut-off_пористость

        Args:
            Кпр_абс(np.ndarray[int/float]): массив с данными абсолютной проницаемости для проверки
            Кп_откр(np.ndarray[int/float]): массив с данными открытой пористости для проверки
            Cut-off_проницаемость(np.ndarray[int/float]): значение критерия отсечения для проницаемости
            Cut-off_пористость(F)(np.ndarray[int/float]): значения критерия отсечения для пористости

        Returns:
            Словарь, specification словарь где, result_mask - словарь с результатом
            {"название параметра/пары параметров":маска с результатом}, test_name - название теста,
            param_name - названия параметров, error_decr -краткое описание ошибки, well_name- название скважины,
            MD - массив с глубинами
        """
        return self.__main_poro_perm_cut_off(self.consts.kp_abs, "cut_off_kp_abs", get_report, filters)
