import inspect

from qa_qc_lib.readers.data_reader import QA_QC_grdecl_parser
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

        attributes = locals()
        for key in attributes.keys():
            if 'file' in key and attributes[key] is not None:
                flag, prop_name = CubesTools().find_key(attributes[key])
                if flag:
                    print(attributes[key], "->", prop_name)
                    self.grid_model.add_prop(attributes[key], prop_name)

    def __generate_report_tests(self, returns_dict: dict, save_path: str = '.', name: str = "QA/QC"):
        CubesTools().generate_wrong_actnum(returns_dict["specification"]["wrong_data"], save_path, name)

    def __generate_returns_dict(self, data_availability: bool, result: bool or None,
                                wrong_data: np.array or None) -> dict:
        wrong_list = None if wrong_data is None else wrong_data.tolist()
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
        value[np.isnan(value)] = 0
        return value

    def __test_range_data(self, _array: np.array, lambda_list: list[any], f) -> tuple[
        bool,
        np.array or None]:

        """
        Абистрактная функция для проверки N условия для np.array

        Args:
            _array (np.array): Данные которые надо проверить
            lambda_list: Массив условий в виде лямдо функций
            f: метод наложения результатов

        Returns:
            tuple[
                bool: Выполняются ли все условия,
                np.array or None: Данные которые не прошли условие
            ]
        """

        mask_array = (f([func(_array) for func in lambda_list])).astype(dtype=bool)
        mask_array = mask_array + (_array == -1)
        if np.all(mask_array):
            return True, None
        else:
            if mask_array.ndim == 1:
                return False, mask_array == False
            else:
                return False, CubesTools().conver_n3d_to_n1d(mask_array == False)

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