import copy
import inspect
import sys

from qa_qc_lib.readers.data_reader import QA_QC_grdecl_parser, QA_QC_asciigrid_parser
from qa_qc_lib.tests.base_test import QA_QC_main
from qa_qc_lib.tools.cubes_tools import CubesTools
import numpy as np
import datetime


class QA_QC_cubes(QA_QC_main):
    def __init__(self,
                 directory_path: str,
                 grid_name: str,
                 qa_qc_kern=None,
                 actnum_file_path: str or None = None,
                 open_porosity_file_path: str or None = None,
                 open_perm_x_file_path: str or None = None,
                 open_perm_y_file_path: str or None = None,
                 open_perm_z_file_path: str or None = None,
                 litatype_file_path: str or None = None,
                 sgcr_file_path: str or None = None,
                 sgl_file_path: str or None = None,
                 sogcr_file_path: str or None = None,
                 sowcr_file_path: str or None = None,
                 sw_file_path: str or None = None,
                 sgu_file_path: str or None = None,
                 swl_file_path: str or None = None,
                 swcr_file_path: str or None = None,
                 swu_file_path: str or None = None,
                 ntg_file_path: str or None = None,
                 so_file_path: str or None = None,
                 sg_file_path: str or None = None,
                 h_abs_ascii_path: str or None = None,
                 h_eff_ascii_path: str or None = None,
                 heffo_ascii_path: str or None = None,
                 heffg_ascii_path: str or None = None,
                 gnk_ascii_path: str or None = None,
                 wnk_ascii_path: str or None = None,
                 save_wrong_data_path: str = ".",
                 ):
        QA_QC_main.__init__(self)
        self.grid_model = QA_QC_grdecl_parser(directory_path, grid_name)
        self.open_porosity_file_path = open_porosity_file_path
        self.open_perm_x_file_path = open_perm_x_file_path
        self.open_perm_y_file_path = open_perm_y_file_path
        self.open_perm_z_file_path = open_perm_z_file_path
        self.litatype_file_path = litatype_file_path
        self.sgcr_file_path = sgcr_file_path
        self.sgl_file_path = sgl_file_path
        self.sogcr_file_path = sogcr_file_path
        self.sowcr_file_path = sowcr_file_path
        self.sw_file_path = sw_file_path
        self.sgu_file_path = sgu_file_path
        self.swl_file_path = swl_file_path
        self.swcr_file_path = swcr_file_path
        self.swu_file_path = swu_file_path
        self.ntg_file_path = ntg_file_path
        self.so_file_path = so_file_path
        self.sg_file_path = sg_file_path
        self.save_wrong_data_path = save_wrong_data_path
        self.h_abs_ascii_path = h_abs_ascii_path
        self.h_eff_ascii_path = h_eff_ascii_path
        self.heffo_ascii_path = heffo_ascii_path
        self.heffg_ascii_path = heffg_ascii_path
        self.gnk_ascii_path = gnk_ascii_path
        self.wnk_ascii_path = wnk_ascii_path

        self.help_dict_type_data = {
            'Porosity': self.open_porosity_file_path,
            'PermX': self.open_perm_x_file_path,
            'PermY': self.open_perm_y_file_path,
            'PermZ': self.open_perm_z_file_path,
            'SGCR': self.sgcr_file_path,
            'SGL': self.sgl_file_path,
            'SOGCR': self.sgcr_file_path,
            'SOWCR': self.sowcr_file_path,
            'SWATINIT': self.sw_file_path,
            'SW': self.sw_file_path,
            'SGU': self.sgu_file_path,
            'SWL': self.swl_file_path,
            'SWCR': self.swcr_file_path,
            'SWU': self.swu_file_path,
            'NTG': self.ntg_file_path,
            'So': self.so_file_path,
            'Sg': self.sg_file_path,
            'Литотип': self.litatype_file_path,
        }

        #Активирует все ячейки грида - mock для проверки
        #self.grid_model.get_grid().activate_all()
        self.actnum = self.grid_model.get_grid().get_actnum().get_npvalues3d()
        self.grid_head = CubesTools().find_head(f"{directory_path}/{grid_name}_ACTNUM.GRDECL")

        attributes = locals()
        for key in attributes.keys():
            if 'file' in key and attributes[key] is not None:
                flag, prop_name = CubesTools().find_key(attributes[key])
                if flag:
                    self.grid_model.add_prop(attributes[key], prop_name)

    def generate_report_tests(self, returns_dict: dict, save_path: str = '.', name: str = "QA-QC"):
        CubesTools().generate_wrong_actnum(np.array(returns_dict["specification"]["wrong_data"]),self.grid_head, save_path, name)

    def __generate_returns_dict(self, data_availability: bool, result: bool or None,
                                wrong_data: np.array or None) -> dict:
        wrong_list = None if wrong_data is None else CubesTools().conver_n3d_to_n1d(wrong_data).tolist()
        return {
            "data_availability": data_availability,
            "result": result,
            "specification": {
                "wrong_data": wrong_list
            }
        }

    def __muc_np_arrays(self, np_array_list):
        """
                Функция перемножения булевых np.array массивов

                Args:
                    np_array_list ([]np.array]): список массивов np.array

                Returns:
                    np.array: результат
        """
        result = np.ones_like(len(np_array_list[0]), 'bool')
        for n in np_array_list:
            result = n * result

        return result

    def __get_value_grid_prop(self, file_path: str, flag_d3: bool = True) -> np.array:
        """
        Функция для получения значения GRDECL файла

        Args:
            file_path (string): Путь к файлу
            flag_d3 (bool): Нужны ли данные в формате куба

        Returns:
            np.array: данные из файла
        """

        _, key = CubesTools().find_key(file_path)
        prop = self.grid_model.get_grid().get_prop_by_name(key)
        value = self.grid_model.get_prop_value(prop, flag_d3)
        return value

    def __test_range_data(self, _array, lambda_list: list[any], f) -> tuple[
        bool,
        np.array or None]:
        """
        Функция для проверки данных в диапазоне

            Args:
               array: список с тестируемыми данными
               lambda_list: list[any]: список с вырожениями для проверки

            Returns:
                bool: результат тестирования
                np.array or None: массив со значениями для wrong actnum
        """
        mask_array = (f([func(_array[self.actnum == 1]) for func in lambda_list])).astype(dtype=bool)
        if np.all(mask_array):
            return True, None
        else:
            copy_actnum = copy.deepcopy(self.grid_model.get_grid().get_actnum())
            copy_actnum.values[np.where(self.actnum == 1)] = (mask_array == False)
            return False, copy_actnum.values


    def __test_value_conditions(self, prop_name: str, lambda_list: list[any], f) -> tuple[
        bool, np.array or None
    ]:
        """
        Абистрактная функция для отправки на проверку конкретных данных из GRDECL файла
        Args:
            prop_name: ключ к данным
            lambda_list: Массив условий в виде лямдо функций
            f: метод наложения результатов

        Returns:
            tuple[
                bool: Выполняются ли все условия,
                np.array or None: Данные которые не прошли условие
            ]
        """

        flag, wrong_data = self.__test_range_data(
            self.grid_model.get_prop_value(
                self.grid_model.get_grid().get_prop_by_name(prop_name),
                False
            ),
            lambda_list=lambda_list,
            f=f)

        if flag:
            return True, None
        else:
            return False, wrong_data

    """
    Тесты первого порядка
    """

    def generate_report_tests_open_porosity(self, returns_dict: dict, save_path: str = '.', name: str = "QA-QC"):
        self.generate_report_tests(returns_dict, save_path, name)

    def test_open_porosity(self) -> dict:
        """
        Функция для проверки открытой пористости

            Required data:
                Porosity;
            Args:
                file_path: str: путь к файлу
                prop_name: str: ключ

            Returns:
                 dict: Словарь, specification cловарь где ,wrong_data - список ячеек куба которые не прошли тестирование

        }
        """
        if self.open_porosity_file_path is None:
            self.update_report(self.generate_report_text("Данные Porosity отсутствуют", 2))
            return self.__generate_returns_dict(False, None, None)
        open_porosity = self.__get_value_grid_prop(self.open_porosity_file_path)
        flag, wrong_data = self.__test_range_data(
            _array=open_porosity,
            lambda_list=[lambda x: x >= 0, lambda x: x <= 0.476],
            f=self.__muc_np_arrays
        )

        if flag:
            r_text = f"Данные лежат в интервале от 0 до 47,6"
            self.update_report(self.generate_report_text(r_text, 1))
            return self.__generate_returns_dict(True, True, None)
        else:
            r_text = f"Данные лежат не в интервале от 0 до 47,6"
            self.update_report(
                self.generate_report_text(
                    r_text,
                    0))

            return self.__generate_returns_dict(True, False, wrong_data)

    def __abstract_test_permeability(self, file_path: str) -> dict:
        """
        Функция для проверки данных на x >= 0

            Args:
                file_path: str: путь к файлу

            Returns:
                 dict: Словарь, specification cловарь где ,wrong_data - список ячеек куба которые не прошли тестирование
        """
        _, key = CubesTools().find_key(file_path)
        flag, wrong_data = self.__test_value_conditions(
            key,
            [lambda x: x >= 0],
            sum
        )

        if flag:
            r_text = f"Данные > 0"
            self.update_report(self.generate_report_text(r_text, 1))
            return self.__generate_returns_dict(True, True, None)
        else:
            r_text = f"Данные < 0"
            self.update_report(self.generate_report_text(
                r_text,
                0))

            return self.__generate_returns_dict(True, False, wrong_data)

    def generate_report_tests_permeability_permX(self, returns_dict: dict, save_path: str = '.',
                                                 name: str = "QA-QC"):
        self.generate_report_tests(returns_dict, save_path, name)

    def test_permeability_permX(self) -> dict:
        """
        Функция для проверки данных на x >= 0

            Required data:
                PermX;

            Returns:
                 dict: Словарь, specification cловарь где ,wrong_data - список ячеек куба которые не прошли тестирование
        """
        if self.open_perm_x_file_path is None:
            return self.__generate_returns_dict(False, None, None)
        return self.__abstract_test_permeability(file_path=self.open_perm_x_file_path)

    def generate_report_tests_permeability_permY(self, returns_dict: dict, save_path: str = '.',
                                                 name: str = "QA-QC"):
        self.generate_report_tests(returns_dict, save_path, name)

    def test_permeability_permY(self) -> dict:
        """
        Функция для проверки данных на x >= 0

            Required data:
                PermY;

            Returns:
                 dict: Словарь, specification cловарь где ,wrong_data - список ячеек куба которые не прошли тестирование
        """
        if self.open_perm_y_file_path is None:
            self.update_report(self.generate_report_text("Данные  PermY отсутствуют", 2))
            return self.__generate_returns_dict(False, None, None)
        return self.__abstract_test_permeability(file_path=self.open_perm_y_file_path)

    def generate_report_tests_permeability_permZ(self, returns_dict: dict, save_path: str = '.',
                                                 name: str = "QA-QC"):
        self.generate_report_tests(returns_dict, save_path, name)

    def test_permeability_permZ(self) -> dict:
        """
        Функция для проверки данных на x >= 0

            Required data:
                PermZ;

            Returns:
                 dict: Словарь, specification cловарь где ,wrong_data - список ячеек куба которые не прошли тестирование
        """
        if self.open_perm_y_file_path is None:
            self.update_report(self.generate_report_text("Данные PermZ отсутствуют", 2))
            return self.__generate_returns_dict(False, None, None)
        return self.__abstract_test_permeability(file_path=self.open_perm_z_file_path)

    def __abstract_test_range_data(self, file_path: str) -> dict:
        """
            Абстрактная функция для проверки данных на x >= 0

            Args:
                file_path: str: путь к файлу

            Returns:
                 dict: Словарь, specification cловарь где ,wrong_data - список ячеек куба которые не прошли тестирование
        """
        _, key = CubesTools().find_key(file_path)

        flag, wrong_data = self.__test_value_conditions(
            key,
            [lambda x: x >= 0, lambda x: x <= 1],
            self.__muc_np_arrays
        )

        if flag:
            r_text = f"Данные лежат в интервале от 0 до 1"
            self.update_report(self.generate_report_text(r_text, 1))
            return self.__generate_returns_dict(True, True, None)
        else:
            r_text = f"Данные лежат не в интервале от 0 до 1"
            print(f"Тест не пройден {r_text}")
            self.update_report(self.generate_report_text(
                r_text,
                0))
            return self.__generate_returns_dict(True, False, wrong_data)

    def generate_report_tests_range_data_sgcr(self, returns_dict: dict, save_path: str = '.', name: str = "QA-QC"):
        self.generate_report_tests(returns_dict, save_path, name)

    def test_range_data_sgcr(self) -> dict:
        """
        Функция для проверки данных на x Є [0:1]

            Required data:
                SGCR;

            Returns:
                 dict: Словарь, specification cловарь где ,wrong_data - список ячеек куба которые не прошли тестирование
        """

        if self.sgcr_file_path is None:
            self.update_report(self.generate_report_text("Данные SGCR отсутствуют", 2))
            return self.__generate_returns_dict(False, None, None)

        return self.__abstract_test_range_data(file_path=self.sgcr_file_path)

    def generate_report_range_data_sgl(self, returns_dict: dict, save_path: str = '.', name: str = "QA-QC"):
        self.generate_report_tests(returns_dict, save_path, name)

    def test_range_data_sgl(self) -> dict:
        """
        Функция для проверки данных на x Є [0:1]

            Required data:
                SGL;

            Returns:
                 dict: Словарь, specification cловарь где ,wrong_data - список ячеек куба которые не прошли тестирование
        """

        if self.sgl_file_path is None:
            self.update_report(self.generate_report_text("Данные SGL отсутствуют", 2))
            return self.__generate_returns_dict(False, None, None)

        return self.__abstract_test_range_data(file_path=self.sgl_file_path)

    def generate_report_range_data_sogcr(self, returns_dict: dict, save_path: str = '.', name: str = "QA-QC"):
        self.generate_report_tests(returns_dict, save_path, name)

    def test_range_data_sogcr(self) -> dict:
        """
        Функция для проверки данных на x Є [0:1]

            Required data:
                SOGCR;

            Returns:
                 dict: Словарь, specification cловарь где ,wrong_data - список ячеек куба которые не прошли тестирование
        """

        if self.sogcr_file_path is None:
            self.update_report(self.generate_report_text("Данные SOGCR отсутствуют", 2))
            return self.__generate_returns_dict(False, None, None)

        return self.__abstract_test_range_data(file_path=self.sogcr_file_path)

    def generate_report_range_data_sowcr(self, returns_dict: dict, save_path: str = '.', name: str = "QA-QC"):
        self.generate_report_tests(returns_dict, save_path, name)

    def test_range_data_sowcr(self) -> dict:
        """
        Функция для проверки данных на x Є [0:1]

            Required data:
                SOWCR;

            Returns:
                 dict: Словарь, specification cловарь где ,wrong_data - список ячеек куба которые не прошли тестирование
        """

        if self.sowcr_file_path is None:
            self.update_report(self.generate_report_text("Данные SOWCR отсутствуют", 2))
            return self.__generate_returns_dict(False, None, None)

        return self.__abstract_test_range_data(file_path=self.sowcr_file_path)

    def generate_report_range_data_swatinit(self, returns_dict: dict, save_path: str = '.', name: str = "QA-QC"):
        self.generate_report_tests(returns_dict, save_path, name)

    def test_range_data_swatinit(self) -> dict:
        """
        Функция для проверки данных на x Є [0:1]

            Required data:
                SWATINIT;

            Returns:
                 dict: Словарь, specification cловарь где ,wrong_data - список ячеек куба которые не прошли тестирование
        """

        if self.sw_file_path is None:
            self.update_report(self.generate_report_text("Данные SWATINIT отсутствуют", 2))
            return self.__generate_returns_dict(False, None, None)

        return self.__abstract_test_range_data(file_path=self.sw_file_path)

    def generate_report_range_data_sgu(self, returns_dict: dict, save_path: str = '.', name: str = "QA-QC"):
        self.generate_report_tests(returns_dict, save_path, name)

    def test_range_data_sgu(self) -> dict:
        """
        Функция для проверки данных на x Є [0:1]

            Required data:
                SGU;

            Returns:
                 dict: Словарь, specification cловарь где ,wrong_data - список ячеек куба которые не прошли тестирование
        """

        if self.sgu_file_path is None:
            self.update_report(self.generate_report_text("Данные SGU отсутствуют", 2))
            return self.__generate_returns_dict(False, None, None)

        return self.__abstract_test_range_data(file_path=self.sgu_file_path)

    def generate_report_range_data_swl(self, returns_dict: dict, save_path: str = '.', name: str = "QA-QC"):
        self.generate_report_tests(returns_dict, save_path, name)

    def test_range_data_swl(self) -> dict:
        """
        Функция для проверки данных на x Є [0:1]

            Required data:
                SWL;

            Returns:
                 dict: Словарь, specification cловарь где ,wrong_data - список ячеек куба которые не прошли тестирование
        """

        if self.swl_file_path is None:
            self.update_report(self.generate_report_text("Данные SWL отсутствуют", 2))
            return self.__generate_returns_dict(False, None, None)

        return self.__abstract_test_range_data(file_path=self.swl_file_path)

    def generate_report_range_data_swcr(self, returns_dict: dict, save_path: str = '.', name: str = "QA-QC"):
        self.generate_report_tests(returns_dict, save_path, name)

    def test_range_data_swcr(self) -> dict:
        """
        Функция для проверки данных на x Є [0:1]

            Required data:
                SWCR;

            Returns:
                 dict: Словарь, specification cловарь где ,wrong_data - список ячеек куба которые не прошли тестирование
        """

        if self.swcr_file_path is None:
            self.update_report(self.generate_report_text("Данные SWCR отсутствуют", 2))
            return self.__generate_returns_dict(False, None, None)

        return self.__abstract_test_range_data(file_path=self.swcr_file_path)

    def generate_report_range_data_swu(self, returns_dict: dict, save_path: str = '.', name: str = "QA-QC"):
        self.generate_report_tests(returns_dict, save_path, name)

    def test_range_data_swu(self) -> dict:
        """
        Функция для проверки данных на x Є [0:1]

            Required data:
                SWU;

            Returns:
                 dict: Словарь, specification cловарь где ,wrong_data - список ячеек куба которые не прошли тестирование
        """

        if self.swu_file_path is None:
            self.update_report(self.generate_report_text("Данные SWU отсутствуют", 2))
            return self.__generate_returns_dict(False, None, None)

        return self.__abstract_test_range_data(file_path=self.swu_file_path)

    def generate_report_range_data_ntg(self, returns_dict: dict, save_path: str = '.', name: str = "QA-QC"):
        self.generate_report_tests(returns_dict, save_path, name)

    def test_range_data_ntg(self) -> dict:
        """
        Функция для проверки данных на x Є [0:1]

            Required data:
                NTG;

            Returns:
                 dict: Словарь, specification cловарь где ,wrong_data - список ячеек куба которые не прошли тестирование
        """

        if self.ntg_file_path is None:
            self.update_report(self.generate_report_text("Данные NTG отсутствуют", 2))
            return self.__generate_returns_dict(False, None, None)

        return self.__abstract_test_range_data(file_path=self.ntg_file_path)

    def generate_report_range_data_so(self, returns_dict: dict, save_path: str = '.', name: str = "QA-QC"):
        self.generate_report_tests(returns_dict, save_path, name)

    def test_range_data_so(self) -> dict:
        """
        Функция для проверки данных на x Є [0:1]

            Required data:
                So;

            Returns:
                 dict: Словарь, specification cловарь где ,wrong_data - список ячеек куба которые не прошли тестирование
        """

        if self.so_file_path is None:
            self.update_report(self.generate_report_text("Данные SO отсутствуют", 2))
            return self.__generate_returns_dict(False, None, None)

        return self.__abstract_test_range_data(file_path=self.so_file_path)

    def generate_report_range_data_sg(self, returns_dict: dict, save_path: str = '.', name: str = "QA-QC"):
        self.generate_report_tests(returns_dict, save_path, name)

    def test_range_data_sg(self) -> dict:
        """
        Функция для проверки данных на x Є [0:1]

            Required data:
                Sg;

            Returns:
                 dict: Словарь, specification cловарь где ,wrong_data - список ячеек куба которые не прошли тестирование
        """

        if self.sg_file_path is None:
            self.update_report(self.generate_report_text("Данные SG отсутствуют", 2))
            return self.__generate_returns_dict(False, None, None)

        return self.__abstract_test_range_data(file_path=self.sg_file_path)

    def generate_report_right_actnum(self, returns_dict: dict, save_path: str = '.', name: str = "QA-QC"):
        self.generate_report_tests(returns_dict, save_path, name)

    def test_right_actnum(self) -> dict:
        """
        Функция для проверки данных на x == 0 || x == 1

            Required data:
                ACTNUM;

            Returns:
                 dict: Словарь, specification cловарь где ,wrong_data - список ячеек куба которые не прошли тестирование
        """
        actnum_array = self.grid_model.get_grid().actnum_array
        if actnum_array is None:
            self.update_report(self.generate_report_text("Данные ACTNUM отсутствуют", 2))
            return self.__generate_returns_dict(False, None, None)
        flag, wrong_data = self.__test_range_data(
            actnum_array,
            [lambda x: x == 0, lambda x: x == 1],
            sum
        )

        if flag:
            r_text = f"Данные равняются 0 или 1"
            self.update_report(self.generate_report_text(r_text, 1))
            return self.__generate_returns_dict(True, True, None)
        else:
            r_text = f"Данные не равняются 0 или 1"
            self.update_report(self.generate_report_text(
                r_text,
                0))

            return self.__generate_returns_dict(True, False, wrong_data)

    def generate_report_litatype(self, returns_dict: dict, save_path: str = '.', name: str = "QA-QC"):
        self.generate_report_tests(returns_dict, save_path, name)

    def test_litatype(self) -> dict:
        """
        Функция для проверки данных на x целое число

            Required data:
                Литотип;

            Returns:
                 dict: Словарь, specification cловарь где ,wrong_data - список ячеек куба которые не прошли тестирование
        """
        if self.litatype_file_path is None:
            self.update_report(self.generate_report_text("Данные LITATYPE отсутствуют", 2))
            return self.__generate_returns_dict(False, None, None)
        _, key = CubesTools().find_key(self.litatype_file_path)
        flag, wrong_data = self.__test_value_conditions(
            key,
            [lambda x: x % 1 == 0],
            sum
        )

        if flag:
            r_text = f"Данные целочисленные"
            self.update_report(self.generate_report_text(r_text, 1))
            return self.__generate_returns_dict(True, True, None)
        else:
            r_text = f"Данные не целочисленные"
            self.update_report(self.generate_report_text(
                r_text,
                0))
            return self.__generate_returns_dict(True, False, wrong_data)

    def generate_report_bulk(self, returns_dict: dict, save_path: str = '.', name: str = "QA-QC"):
        self.generate_report_tests(returns_dict, save_path, name)

    def test_bulk(self):
        """
        Функция для проверки данных геометрического объема grid-a, должен быть не отрицательным

            Returns:
                     dict: Словарь, specification cловарь где ,wrong_data - список ячеек куба которые не прошли тестирование
        """
        data = self.grid_model.get_grid().get_bulk_volume(asmasked=False).get_npvalues3d()
        data[np.isnan(data)] = 0

        flag, wrong_data = self.__test_range_data(
            data,
            [lambda x: x >= 0],
            sum
        )

        if flag:
            r_text = f"Данные имеют положительный объем"
            self.update_report(self.generate_report_text(r_text, 1))
            return self.__generate_returns_dict(True, True, None)
        else:
            r_text = f"Данные имеют отрицательный объем"
            self.update_report(self.generate_report_text(
                r_text,
                0))
            return self.__generate_returns_dict(True, False, wrong_data)

    """
    Тесты второго порядка
    """
    def generate_report_sum(self, returns_dict: dict, save_path: str = '.', name: str = "QA-QC"):
        self.generate_report_tests(returns_dict, save_path, name)
    def test_sum_cubes(self):
        """
        Функция для проверки того что сумма кубов = 1

            Required data:
                SWATINIT;
                Sg;
                So;

            Returns:
                 dict: Словарь, specification cловарь где ,wrong_data - список ячеек куба которые не прошли тестирование
        """
        if self.sw_file_path is None:
            self.update_report(self.generate_report_text("Данные SWATINIT отсутствуют", 2))
            return self.__generate_returns_dict(False, None, None)
        if self.so_file_path is None:
            self.update_report(self.generate_report_text("Данные So отсутствуют", 2))
            return self.__generate_returns_dict(False, None, None)
        if self.sg_file_path is None:
            self.update_report(self.generate_report_text("Данные Sg отсутствуют", 2))
            return self.__generate_returns_dict(False, None, None)

        mas_file_path = [self.sw_file_path, self.so_file_path]
        if self.sg_file_path is not None:
            mas_file_path.append(self.sg_file_path)

        data_mas = []
        for file_path in mas_file_path:
            _, prop_name = CubesTools().find_key(
                file_path)
            data_mas.append(
                self.grid_model.get_prop_value(
                    self.grid_model.get_grid().get_prop_by_name(
                        prop_name
                    )
                )
            )

        flag, wrong_data = self.__test_range_data(
            sum(data_mas),
            [lambda x: x >= 1-0.02, lambda x: x <= 1],
            self.__muc_np_arrays
        )

        if flag:
            r_text = f"Cумма кубов = 1"
            self.update_report(self.generate_report_text(r_text, 1))
            return self.__generate_returns_dict(True, True, None)
        else:
            r_text = f"Cумма кубов != 1"
            print(f"Тест не пройден {r_text}")
            self.update_report(self.generate_report_text(
                r_text,
                0))
            return self.__generate_returns_dict(True, False, wrong_data)

    def test_affiliation_sgcr(self):
        """
        Функция для проверки того что SGCR Є [SGL:1]

            Required data:
                SGCR;
                SGL;

            Returns:
                 dict: Словарь, specification cловарь где ,wrong_data - список ячеек куба которые не прошли тестирование
        """

        if self.sgcr_file_path is None:
            self.update_report(self.generate_report_text("Данные SGCR отсутствуют", 2))
            return self.__generate_returns_dict(False, None, None)
        if self.sgl_file_path is None:
            self.update_report(self.generate_report_text("Данные SGL отсутствуют", 2))
            return self.__generate_returns_dict(False, None, None)

        sgcr_value = self.__get_value_grid_prop(self.sgcr_file_path)
        sgl_value = self.__get_value_grid_prop(self.sgl_file_path)

        flag, wrong_data = self.__test_range_data(
            sgcr_value,
            [lambda x: x >= sgl_value[self.actnum == 1], lambda x: x <= 1],
            self.__muc_np_arrays
        )

        if flag:
            r_text = f"Данные  U [SGL:1] "
            self.update_report(self.generate_report_text(r_text, 1))
            return self.__generate_returns_dict(True, True, None)
        else:
            r_text = f"Данные  ∉ [SGL:1] "
            self.update_report(self.generate_report_text(
                r_text,
                0))
            return self.__generate_returns_dict(True, False, wrong_data)

    def test_affiliation_swcr(self):
        """
        Функция для проверки того что SWCR ≥ SWL

            Required data:
                SWCR;
                SWL;

            Returns:
                 dict: Словарь, specification cловарь где ,wrong_data - список ячеек куба которые не прошли тестирование
        """
        if self.swcr_file_path is None:
            self.update_report(self.generate_report_text("Данные SWCR отсутствуют", 2))
            return self.__generate_returns_dict(False, None, None)
        if self.swl_file_path is None:
            self.update_report(self.generate_report_text("Данные SWL отсутствуют", 2))
            return self.__generate_returns_dict(False, None, None)

        swcr_value = self.__get_value_grid_prop(self.swcr_file_path)

        swl_value = self.__get_value_grid_prop(self.swl_file_path)

        flag, wrong_data = self.__test_range_data(
            swcr_value,
            [lambda x: x >= swl_value[self.actnum == 1]],
            sum
        )

        if flag:
            r_text = f"Данные >= SWL "
            self.update_report(self.generate_report_text(r_text, 1))
            return self.__generate_returns_dict(True, True, None)
        else:
            r_text = f"Данные < SWL "
            self.update_report(self.generate_report_text(
                r_text,
               0))

            return self.__generate_returns_dict(True, False, wrong_data)

    def test_swl_sw(self):
        """
        Проверка SWATINIT >= SWL

            Required data:
                SWATINIT
                SWL

            Returns:
                 dict: Словарь, specification cловарь где ,wrong_data - список ячеек куба которые не прошли тестирование
        """
        if self.swl_file_path is None:
            self.update_report(self.generate_report_text("Данные SWL отсутствуют", 2))
            return self.__generate_returns_dict(False, None, None)
        if self.sw_file_path is None:
            self.update_report(self.generate_report_text("Данные SWATINIT отсутствуют", 2))
            return self.__generate_returns_dict(False, None, None)

        swl_value = self.__get_value_grid_prop(self.swl_file_path)
        sw_value = self.__get_value_grid_prop(self.sw_file_path)
        flag, wrong_data = self.__test_range_data(
            sw_value,
            [lambda x: x >= swl_value[self.actnum == 1]],
            sum
        )

        if flag:
            r_text = f"Данные > SWL "
            self.update_report(self.generate_report_text(r_text, 1))
            return self.__generate_returns_dict(True, True, None)
        else:
            r_text = f"Данные <= SWL "
            self.update_report(
                self.generate_report_text(
                    r_text,
                    0))

            return self.__generate_returns_dict(True, False, wrong_data)

    def test_inconsistencies_habs_heff(self):
        """
        При сравнении двух карт, в каждой в ij-точке значение абсолютной толщины должно быть больше или равно значению эффективной толщины:
        Набс ≥ Нэфф


            Required data:
                Абсолютные_толщины(h_abs)
                Эффективные_толщины(h_eff)

            Returns:
                 dict: Словарь, specification cловарь где ,wrong_data - список ячеек куба которые не прошли тестирование
        """
        if self.h_abs_ascii_path is None:
            self.update_report(self.generate_report_text("Данные H_ABS отсутствуют", 2))
            return self.__generate_returns_dict(False, None, None)

        if self.h_eff_ascii_path is None:
            self.update_report(self.generate_report_text("Данные H_EFF отсутствуют", 2))
            return self.__generate_returns_dict(False, None, None)

        h_abs_data, head = QA_QC_asciigrid_parser().parse_to_dataframe(self.h_abs_ascii_path)
        h_eff_data, _ = QA_QC_asciigrid_parser().parse_to_dataframe(self.h_eff_ascii_path)

        flag, wrong_data = self.__test_range_data(
            h_abs_data['z'],
            [lambda  x: x >= h_eff_data['z']],
            sum
        )

        if flag:
            r_text = f"Данные H_ABS >= H_EFF"
            self.update_report(self.generate_report_text(r_text, 1))
            return self.__generate_returns_dict(True, True, None)
        else:
            r_text = f"Данные H_ABS < H_EFF"
            self.update_report(
                self.generate_report_text(
                    r_text,
                    0))

            return self.__generate_returns_dict(True, False, wrong_data)

    def test_inconsistencies_habs_heff_heffo(self):
        """
        При сопоставлении трех карт в каждой в ij-точке значение абсолютной толщины должно быть больше или равно значению эффективной толщины, которое в свою очередь больше или равно значению эффективной нефтенасыщенной толщины
        Набс ≥ Нэфф ≥ Нннт


            Required data:
                Абсолютные_толщины(h_abs)
                Эффективные_толщины(h_eff)
                Эффективные_нефтенасыщенные_толщины(heffo)

            Returns:
                 dict: Словарь, specification cловарь где ,wrong_data - список ячеек куба которые не прошли тестирование
        """
        if self.h_abs_ascii_path is None:
            self.update_report(self.generate_report_text("Данные H_ABS отсутствуют", 2))
            return self.__generate_returns_dict(False, None, None)

        if self.h_eff_ascii_path is None:
            self.update_report(self.generate_report_text("Данные H_EFF отсутствуют", 2))
            return self.__generate_returns_dict(False, None, None)

        if self.heffo_ascii_path is None:
            self.update_report(self.generate_report_text("Данные HEFFO отсутствуют", 2))
            return self.__generate_returns_dict(False, None, None)

        h_abs_data, head = QA_QC_asciigrid_parser().parse_to_dataframe(self.h_abs_ascii_path)
        h_eff_data, _ = QA_QC_asciigrid_parser().parse_to_dataframe(self.h_eff_ascii_path)
        heffo_data, _ = QA_QC_asciigrid_parser().parse_to_dataframe(self.heffo_ascii_path)

        flag, wrong_data = self.__test_range_data(
            h_abs_data,
            [lambda x: x >= h_eff_data],
            sum
        )

        if flag:
            flag, wrong_data = self.__test_range_data(
                h_eff_data['z'],
                [lambda x: x >= heffo_data['z']],
                sum
            )

            if flag:
                r_text = f"Данные H_EFF >= HEFFO"
                self.update_report(self.generate_report_text(r_text, 1))
                return self.__generate_returns_dict(True, True, None)
            else:
                r_text = f"Данные H_EFF < HEFFO"
                self.update_report(
                    self.generate_report_text(
                        r_text,
                        0))

                return self.__generate_returns_dict(True, False, wrong_data)
        else:
            r_text = f"Данные H_ABS < H_EFF"
            self.update_report(
                self.generate_report_text(
                    r_text,
                    0))

            return self.__generate_returns_dict(True, False, wrong_data)

    def test_inconsistencies_habs_heff_heffg(self):
        """
        При сопоставлении трех карт в каждой в ij-точке значение абсолютной толщины должно быть больше или равно значению эффективной толщины, которое в свою очередь больше или равно значению эффективной газонасыщенной толщины
        Набс ≥ Нэфф ≥ Нгнт


            Required data:
                Абсолютные_толщины(h_abs)
                Эффективные_толщины(h_eff)
                Эффективные_газонасыщенные_толщины(heffg)

            Returns:
                 dict: Словарь, specification cловарь где ,wrong_data - список ячеек куба которые не прошли тестирование
        """
        if self.h_abs_ascii_path is None:
            self.update_report(self.generate_report_text("Данные H_ABS отсутствуют", 2))
            return self.__generate_returns_dict(False, None, None)

        if self.h_eff_ascii_path is None:
            self.update_report(self.generate_report_text("Данные H_EFF отсутствуют", 2))
            return self.__generate_returns_dict(False, None, None)

        if self.heffg_ascii_path is None:
            self.update_report(self.generate_report_text("Данные HEFFG отсутствуют", 2))
            return self.__generate_returns_dict(False, None, None)

        h_abs_data, head = QA_QC_asciigrid_parser().parse_to_dataframe(self.h_abs_ascii_path)
        h_eff_data, _ = QA_QC_asciigrid_parser().parse_to_dataframe(self.h_eff_ascii_path)
        heffg_data, _ = QA_QC_asciigrid_parser().parse_to_dataframe(self.heffg_ascii_path)

        flag, wrong_data = self.__test_range_data(
            h_abs_data['z'],
            [lambda x: x >= h_eff_data['z']],
            sum
        )

        if flag:
            flag, wrong_data = self.__test_range_data(
                h_eff_data['z'],
                [lambda x: x >= heffg_data['z']],
                sum
            )

            if flag:
                r_text = f"Данные H_EFF >= HEFFG"
                self.update_report(self.generate_report_text(r_text, 1))
                return self.__generate_returns_dict(True, True, None)
            else:
                r_text = f"Данные H_EFF < HEFFG"
                self.update_report(
                    self.generate_report_text(
                        r_text,
                        0))

                return self.__generate_returns_dict(True, False, wrong_data)
        else:
            r_text = f"Данные H_ABS < H_EFF"
            self.update_report(
                self.generate_report_text(
                    r_text,
                    0))

            return self.__generate_returns_dict(True, False, wrong_data)

    def test_incorrect_values_sgu(self):


        if self.gnk_ascii_path is None:
            self.update_report(self.generate_report_text("Данные ГНК отсутствуют", 2))
            return self.__generate_returns_dict(False, None, None)

        #df,head = QA_QC_asciigrid_parser().parse_to_dataframe(self.gnk_ascii_path)
        #pol = QA_QC_asciigrid_parser().get_pologin(df)
        #self.grid_model.get_grid().activate_all()
        #self.grid_model.get_grid().inactivate_outside(pol)
        act = self.grid_model.get_grid().get_actnum().get_npvalues3d()
        print(len(act[0]))