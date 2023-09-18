from qa_qc_lib.data_reader import QA_QC_grdecl_parser
from qa_qc_lib.qa_qc_main import QA_QC_main, Type_Status
import numpy as np
import datetime


class QA_QC_cubes(QA_QC_main):

    def __init__(self, directory_path: str, grid_name: str):
        QA_QC_main.__init__(self)
        self.grid_model = QA_QC_grdecl_parser(directory_path, grid_name)

    def __pars_wrong_values(self, wrong_list: list[
        list[int or float],
        list[int]
    ]) -> str:
        """
        Функция для распарса данных в строку для результата

            Args:
                wrong_list: list[
                    list[int or float],
                    list[int]: массив со значениями и их ijz индексами

            Returns:
                str: строка с данными
        """
        result = sum([
            f"value:{wrong_list[0][i]} " + f"index: x={wrong_list[1][i][0]}, y={wrong_list[1][i][1]}, z={wrong_list[1][i][2]}\n"
            for i in range(len(wrong_list[0]))])
        return result

    def __test_range_data(self, array, lambda_list: list[any]) -> tuple[
        bool,
        list[
            list[int or float],
            list[int]
        ] or None]:
        """
        Функция для проверки данных в диапазоне

            Args:
               array: список с тестируемыми данными
               lambda_list: list[any]: список с вырожениями для проверки

            Returns:
                bool: результат тестирования
                list[list[int/float],list[int]] or None: массив со значениями и их ijz индексами
        """
        mask_array = (sum(func(array) for func in lambda_list)).astype(dtype=bool)

        if not all(mask_array):
            return True, None
        else:
            return False, [array[mask_array == False].tolist(), np.argwhere(mask_array == False)[0].tolist()]

    def __test_value_conditions(self, file_path: str, prop_name: str, lambda_list: list[any]) -> tuple[
        bool,
        str or None
    ]:
        """
        Функция для парса данных и отправки их на проверку

            Args:
                file_path: str: путь к файлу
                prop_name: str: ключ
                lambda_list: list[any]: список с вырожениями для проверки

            Returns:
                bool: результат тестирования
                str or None: страка с результатом
        """
        self.grid_model.add_prop(file_path, prop_name)

        flag, wrong_data = self.__test_range_data(
            self.grid_model.get_prop_value(self.grid_model.get_grid().get_prop_by_name(prop_name)),
            lambda_list)

        if flag:
            return True, ""
        else:
            return False, self.__pars_wrong_values(wrong_data)

    """
    Тесты первого порядка
    """

    def test_open_porosity(self, file_path: str, prop_name: str):
        """
        Функция для проверки открытой пористости

            Required data:
                Porosity|GRDECL|ПЕТРОФИЗИКА|;
            Args:
                file_path: str: путь к файлу
                prop_name: str: ключ
        """
        flag, wrong_data = self.__test_value_conditions(
            file_path,
            prop_name,
            [lambda x: x > 0, lambda x: x <= 0.476]
        )

        if flag:
            self.generate_report_text("", Type_Status.Passed.value)
        else:
            self.generate_report_text(f"Данные \n {wrong_data}  лежат не в интервале от 0 до 47,6",
                                      Type_Status.NotPassed)

    def test_permeability(self, file_path: str, prop_name: str):
        """
        Функция для проверки данных на x >= 0

            Required data:
                PermX|GRDECL|ПЕТРОФИЗИКА|;
                PermY|GRDECL|ПЕТРОФИЗИКА|;
                PermZ|GRDECL|ПЕТРОФИЗИКА|;
                J-function|GRDECL|ПЕТРОФИЗИКА|;
            Args:
                file_path: str: путь к файлу
                prop_name: str: ключ
        """
        flag, wrong_data = self.__test_value_conditions(
            file_path,
            prop_name
            [lambda x: x >= 0]
        )

        if flag:
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
        else:
            self.update_report(self.generate_report_text(
                f"Данные \n {wrong_data}  < 0",
                Type_Status.NotPassed))

    def test_range_data(self, file_path: str, prop_name: str):
        """
        Функция для проверки данных на x Є [0:1]

            Required data:
                SGCR|GRDECL|ПЕТРОФИЗИКА|;
                SGL|GRDECL|ПЕТРОФИЗИКА|;
                SOGCR|GRDECL|ПЕТРОФИЗИКА|;
                SOWCR|GRDECL|ПЕТРОФИЗИКА|;
                SWATINIT|GRDECL|ПЕТРОФИЗИКА|;
                SGU|GRDECL|ПЕТРОФИЗИКА|;
                SWL|GRDECL|ПЕТРОФИЗИКА|;
                SWCR|GRDECL|ПЕТРОФИЗИКА|;
                SWU|GRDECL|ПЕТРОФИЗИКА|;
                NTG|GRDECL|ПЕТРОФИЗИКА|;
                So|GRDECL|ПЕТРОФИЗИКА|;
                Sg|GRDECL|ПЕТРОФИЗИКА|;

            Args:
                file_path: str: путь к файлу
                prop_name: str: ключ
        """
        flag, wrong_data = self.__test_value_conditions(
            file_path,
            prop_name
            [lambda x: x >= 0, lambda x: x <= 1]
        )

        if flag:
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
        else:
            self.update_report(self.generate_report_text(
                f"Данные \n {wrong_data}  лежат не в интервале от 0 до 1 ",
                Type_Status.NotPassed))

    def test_range_integer_data(self, file_path: str, prop_name: str):
        """
        Функция для проверки данных на x == 0 || x == 1

            Required data:
                ACTNUM|GRDECL|ПЕТРОФИЗИКА|;

            Args:
                file_path: str: путь к файлу
                prop_name: str: ключ
        """
        flag, wrong_data = self.__test_value_conditions(
            file_path,
            prop_name
            [lambda x: x == 0, lambda x: x == 1]
        )

        if flag:
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
        else:
            self.update_report(self.generate_report_text(
                f"Данные \n {wrong_data}  не равняются 0 или 1",
                Type_Status.NotPassed))

    def test_integer_data(self, file_path: str, prop_name: str):
        """
        Функция для проверки данных на x целое число

            Required data:
                Литотип|GRDECL|ПЕТРОФИЗИКА|;

            Args:
                file_path: str: путь к файлу
                prop_name: str: ключ
        """
        flag, wrong_data = self.__test_value_conditions(
            file_path,
            prop_name
            [lambda x: x % 1 != 0]
        )

        if flag:
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
        else:
            self.update_report(self.generate_report_text(
                f"Данные \n {wrong_data}  не целочисленные",
                Type_Status.NotPassed))

    def test_bulk(self):
        """
        Функция для проверки данных геометрического объема grid-a, должен быть не отрицательным

            Required data:
                GRID_модели|GRDECL|Сейсмика|;
        """
        data = self.grid_model.get_grid().get_bulk_volume(asmasked=False)
        data[np.isnan(data)] = 0

        flag, wrong_data = self.__test_range_data(
            data,
            [lambda x: x >= 0]
        )

        if flag:
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
        else:
            self.update_report(self.generate_report_text(
                f"Данные \n {wrong_data}  имеют отрицательный объем",
                Type_Status.NotPassed))

    """
    Тесты второго порядка
    """

    def test_sum_cubes(self, dict_file_path: dict[str:str]):
        """
        Функция для проверки того что сумма кубов = 1

            Required data:
                SWATINIT|GRDECL|ПЕТРОФИЗИКА|;
                Sg|GRDECL|ПЕТРОФИЗИКА|;
                So|GRDECL|ПЕТРОФИЗИКА|;

            Args:
                dict_file_path: dict[str:str]: словарь {ключ: путь к файлу}
        """
        data_mas = []
        for name_prop in dict_file_path.keys():
            self.grid_model.add_prop(dict_file_path[name_prop], name_prop)
            data_mas.append(self.grid_model.get_prop_value(self.grid_model.get_grid().get_prop_by_name(name_prop)))

        flag, wrong_data = self.__test_range_data(
            data_mas,
            [lambda x: sum(x) == 1]
        )

        if flag:
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
        else:
            self.update_report(self.generate_report_text(
                f"Данные \n {wrong_data}  сумма != 1",
                Type_Status.NotPassed))

    def test_affiliation_sqcr(self, sgcr_path: str, sgl_path: str):
        """
        Функция для проверки того что SGCR Є [SGL:1]

            Required data:
                SGCR|GRDECL|ПЕТРОФИЗИКА|;
                SGL|GRDECL|ПЕТРОФИЗИКА|;

            Args:
                sgcr_path: str: путь к файлу
                sgl_path: str: путь к файлу
        """
        self.grid_model.add_prop(sgcr_path, "SGCR")
        self.grid_model.add_prop(sgl_path, "IRR.GASSATURATION")

        sgcr_value = self.grid_model.get_grid().get_prop_by_name("SGCR")
        sgcr_value[np.isnan(sgcr_value)] = 0

        sgl_value = self.grid_model.get_grid().get_prop_by_name("IRR.GASSATURATION")
        sgl_value[np.isnan(sgl_value)] = 0

        flag, wrong_data = self.__test_range_data(
            sgcr_value,
            [lambda x: x >= sgl_value, lambda x: x <= 1]
        )

        if flag:
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
        else:
            self.update_report(self.generate_report_text(
                f"Данные \n {wrong_data}  ∉ [SGL:1] ",
                Type_Status.NotPassed))

    def test_affiliation_swcr(self, swcr_path: str, swl_path: str):
        """
        Функция для проверки того что SWCR ≥ SWL

            Required data:
                SGCR|GRDECL|ПЕТРОФИЗИКА|;
                SGL|GRDECL|ПЕТРОФИЗИКА|;

            Args:
                swcr_path: str: путь к файлу
                swl_path: str: путь к файлу
        """
        self.grid_model.add_prop(swcr_path, "CRITICALWATERSATURATION")
        self.grid_model.add_prop(swl_path, "IRR.WATERSATURATION")

        swcr_value = self.grid_model.get_grid().get_prop_by_name("CRITICALWATERSATURATION")
        swcr_value[np.isnan(swcr_value)] = 0

        swl_value = self.grid_model.get_grid().get_prop_by_name("IRR.WATERSATURATION")
        swl_value[np.isnan(swl_value)] = 0

        flag, wrong_data = self.__test_range_data(
            swcr_value,
            [lambda x: x >= swl_value]
        )

        if flag:
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
        else:
            self.update_report(self.generate_report_text(
                f"Данные \n {wrong_data}  < SWL ",
                Type_Status.NotPassed))

    def test_porosity_value(self, porosity_path: str, cut_off_porosity: float):
        """
        Функция для проверки того что Если в ячейке куба пористости, значение пористости меньше, чем cut-off, то куб ACTNUM = 0

            Required data:
                Porosity|GRDECL|ПЕТРОФИЗИКА|;
                ACTNUM|GRDECL|ПЕТРОФИЗИКА|;
                Cut-off_пористость|txt/xlsx|Керн|

            Args:
                porosity_path: str: путь к файлу
                cut_off_porosity: float: значение Cut-off_пористость
        """
        self.grid_model.add_prop(porosity_path, "PORO")
        poro_value = self.grid_model.get_grid().get_prop_by_name("PORO")
        poro_value[np.isnan(poro_value)] = 0

        flag, wrong_data = self.__test_range_data(
            self.grid_model.get_grid().actnum_array,
            [lambda x: x[poro_value < cut_off_porosity] == 0]
        )

        if flag:
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
        else:
            self.update_report(self.generate_report_text(
                f"Данные \n {wrong_data}  значение porosity < cut-off porosity, но ACTNUM == 1 ",
                Type_Status.NotPassed))


# self.generate_test_report(file_name=self.__class__.__name__, file_path="../../report", data_name=file_path)


#QA_QC_cubes("../../data/grdecl_data/GRID.GRDECL").test_open_porosity("../../data/grdecl_data/input/Poro.GRDECL.grdecl")
