import inspect

from qa_qc_lib.data_reader import QA_QC_grdecl_parser
from qa_qc_lib.qa_qc_main import QA_QC_main, Type_Status
from qa_qc_lib.qa_qc_tools.cubes_tools import CubesTools
from qa_qc_lib.qa_qc_connectors.connector_kern_cubes import Connector_kern_cubes
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
        return {
            "data_availability": data_availability,
            "result": result,
            "specification": {
                "wrong_data": wrong_data
            }
        }

    def generate_report_text(self, text, status):
        status_dict = {0: 'Тест не пройден.',
                       1: 'Тест пройден успешно.',
                       2: 'Тест не был запущен.'}
        report_text = f"{self.ident}{status_dict[status]}\n{self.ident}{text}"
        return report_text

    def __muc_np_arrays(self, np_array_list):
        result = np.ones_like(len(np_array_list[0]), 'bool')
        for n in np_array_list:
            result = n * result

        return result

    def __get_value_grid_prop(self, file_path: str, flag_d3: bool = True) -> np.array:
        _, key = CubesTools().find_key(file_path)
        prop = self.grid_model.get_grid().get_prop_by_name(key)
        value = self.grid_model.get_prop_value(prop, flag_d3)
        value[np.isnan(value)] = 0
        return value

    def __test_range_data(self, _array, lambda_list: list[any], f) -> tuple[
        bool,
        np.array or None]:

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

    def test_open_porosity(self) -> dict:
        """
        Функция для проверки открытой пористости

            Required data:
                Porosity;
            Args:
                file_path: str: путь к файлу
                prop_name: str: ключ
        """
        if self.open_porosity_file_path is None:
            return self.__generate_returns_dict(False, None, None)
        _, key = CubesTools().find_key(self.open_porosity_file_path)
        flag, wrong_data = self.__test_value_conditions(
            prop_name=key,
            lambda_list=[lambda x: x >= 0, lambda x: x <= 0.476],
            f=self.__muc_np_arrays
        )

        if flag:
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
            return self.__generate_returns_dict(True, True, None)
        else:
            r_text = f"Данные лежат не в интервале от 0 до 47,6"
            self.update_report(self.generate_report_text(f"Данные лежат не в интервале от 0 до 47,6",
                                                         Type_Status.NotPassed.value))

            return self.__generate_returns_dict(True, False, wrong_data)

    def test_permeability(self, name_type_data: str) -> dict:
        """
        Функция для проверки данных на x >= 0

            Required data:
                PermX;
                PermY;
                PermZ;
                J-function;
            Args:
                file_path: str: путь к файлу
                prop_name: str: ключ
        """
        if not (name_type_data in self.help_dict_type_data):
            print("Названия типа данных не существует")
            self.update_report(
                self.generate_report_text("Названия типа данных не существует", Type_Status.NotRunning.value))
            return self.__generate_returns_dict(False, None, None)
        _, key = CubesTools().find_key(self.help_dict_type_data[name_type_data])
        flag, wrong_data = self.__test_value_conditions(
            key,
            [lambda x: x >= 0],
            sum
        )

        if flag:
            print("Тест пройден")
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
            return self.__generate_returns_dict(True, True, None)
        else:
            r_text = f"Данные < 0"
            print(f"Тест не пройден {r_text}")
            self.update_report(self.generate_report_text(
                r_text,
                Type_Status.NotPassed.value))

            return self.__generate_returns_dict(True, False, wrong_data)

    def test_range_data(self, name_type_data: str) -> dict:
        """
        Функция для проверки данных на x Є [0:1]

            Required data:
                SGCR;
                SGL;
                SOGCR;
                SOWCR;
                SWATINIT;
                SGU;
                SWL;
                SWCR;
                SWU;
                NTG;
                So;
                Sg;

            Args:
                file_path: str: путь к файлу
                prop_name: str: ключ
        """
        if not (name_type_data in self.help_dict_type_data):
            print("Названия типа данных не существует")
            self.update_report(
                self.generate_report_text("Названия типа данных не существует", Type_Status.NotRunning.value))
            return self.__generate_returns_dict(False, None, None)

        _, key = CubesTools().find_key(self.help_dict_type_data[name_type_data])

        flag, wrong_data = self.__test_value_conditions(
            key,
            [lambda x: x >= 0, lambda x: x <= 1],
            self.__muc_np_arrays
        )

        if flag:
            print("Тест пройден")
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
            return self.__generate_returns_dict(True, True, None)
        else:
            r_text = f"Данные лежат не в интервале от 0 до 1"
            print(f"Тест не пройден {r_text}")
            self.update_report(self.generate_report_text(
                r_text,
                Type_Status.NotPassed.value))
            return self.__generate_returns_dict(True, False, wrong_data)

    def test_range_integer_data(self) -> dict:
        """
        Функция для проверки данных на x == 0 || x == 1

            Required data:
                ACTNUM;

            Args:
                file_path: str: путь к файлу
                prop_name: str: ключ
        """
        actnum_array = self.grid_model.get_grid().actnum_array
        flag, wrong_data = self.__test_range_data(
            actnum_array,
            [lambda x: x == 0, lambda x: x == 1],
            sum
        )

        if flag:
            print("Тест пройден")
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
            return self.__generate_returns_dict(True, True, None)
        else:
            CubesTools().generate_wrong_actnum(wrong_data, self.save_wrong_data_path, "test_range_integer_data")
            r_text = f"Данные не равняются 0 или 1"
            print(f"Тест не пройден {r_text}")
            self.update_report(self.generate_report_text(
                f"Данные не равняются 0 или 1",
                Type_Status.NotPassed.value))

            return self.__generate_returns_dict(True, False, wrong_data)

    def test_integer_data(self) -> dict:
        """
        Функция для проверки данных на x целое число

            Required data:
                Литотип;

            Args:
                file_path: str: путь к файлу
                prop_name: str: ключ
        """
        if self.litatype_file_path is None:
            return self.__generate_returns_dict(False, None, None)
        _, key = CubesTools().find_key(self.litatype_file_path)
        flag, wrong_data = self.__test_value_conditions(
            key,
            [lambda x: x % 1 == 0],
            sum
        )

        if flag:
            print("Тест пройден")
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
            return self.__generate_returns_dict(True, True, None)
        else:
            r_text = f"Данные не целочисленные"
            print(f"Тест не пройден {r_text}")
            self.update_report(self.generate_report_text(
                r_text,
                Type_Status.NotPassed.value))
            return self.__generate_returns_dict(True, False, wrong_data)

    def test_bulk(self):
        """
        Функция для проверки данных геометрического объема grid-a, должен быть не отрицательным
        """
        data = self.grid_model.get_grid().get_bulk_volume(asmasked=False).get_npvalues3d()
        data[np.isnan(data)] = 0

        flag, wrong_data = self.__test_range_data(
            data,
            [lambda x: x >= 0],
            sum
        )

        if flag:
            print("Тест пройден")
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
            return self.__generate_returns_dict(True, True, None)
        else:
            r_text = f"Данные имеют отрицательный объем"
            print(f"Тест не пройден {r_text}")
            self.update_report(self.generate_report_text(
                r_text,
                Type_Status.NotPassed.value))
            return self.__generate_returns_dict(True, False, wrong_data)

    """
    Тесты второго порядка
    """

    def test_sum_cubes(self, list_name_type_file: list[str]) -> dict:
        """
        Функция для проверки того что сумма кубов = 1

            Required data:
                SWATINIT;
                Sg;
                So;

            Args:
                list_name_type_file: list[str]: список имет типов файлов
        """

        data_mas = []
        for name_type_file in list_name_type_file:
            _, prop_name = CubesTools().find_key(
                self.help_dict_type_data[name_type_file])
            data_mas.append(
                self.grid_model.get_prop_value(
                    self.grid_model.get_grid().get_prop_by_name(
                        prop_name
                    )
                )
            )

        flag, wrong_data = self.__test_range_data(
            data_mas,
            [lambda x: sum(x) == 1, lambda x: sum(x) == 0],
            sum
        )

        if flag:
            print("Тест пройден")
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
            return self.__generate_returns_dict(True, True, None)
        else:
            r_text = f"Cумма кубов != 1"
            print(f"Тест не пройден {r_text}")
            self.update_report(self.generate_report_text(
                r_text,
                Type_Status.NotPassed.value))

            return self.__generate_returns_dict(True, False, wrong_data)

    def test_affiliation_sqcr(self):
        """
        Функция для проверки того что SGCR Є [SGL:1]

            Required data:
                SGCR;
                SGL;

            Args:
                sgcr_path: str: путь к файлу
                sgl_path: str: путь к файлу
        """

        try:
            sgcr_value = self.__get_value_grid_prop(self.sgcr_file_path)
            sgl_value = self.__get_value_grid_prop(self.sgl_file_path)
        except:
            return self.__generate_returns_dict(False, None, None)
        flag, wrong_data = self.__test_range_data(
            sgcr_value,
            [lambda x: x >= sgl_value, lambda x: x <= 1],
            self.__muc_np_arrays
        )

        if flag:
            print("Тест пройден")
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
            return self.__generate_returns_dict(True, True, None)
        else:
            r_text = f"Данные  ∉ [SGL:1] "
            print(f"Тест не пройден {r_text}")
            self.update_report(self.generate_report_text(
                r_text,
                Type_Status.NotPassed.value))

            return self.__generate_returns_dict(True, False, wrong_data)

    def test_affiliation_swcr(self) -> dict:
        """
        Функция для проверки того что SWCR ≥ SWL

            Required data:
                SGCR;
                SGL;

            Args:
                swcr_path: str: путь к файлу
                swl_path: str: путь к файлу
        """
        try:
            swcr_value = self.__get_value_grid_prop(self.swcr_file_path)

            swl_value = self.__get_value_grid_prop(self.swl_file_path)

        except:
            return self.__generate_returns_dict(False, None, None)
        flag, wrong_data = self.__test_range_data(
            swcr_value,
            [lambda x: x >= swl_value],
            sum
        )

        if flag:
            print("Тест пройден")
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
            return self.__generate_returns_dict(True, True, None)
        else:
            r_text = f"Данные < SWL "
            print(f"Тест не пройден {r_text}")
            self.update_report(self.generate_report_text(
                r_text,
                Type_Status.NotPassed.value))

            return self.__generate_returns_dict(True, False, wrong_data)

    def test_porosity_value(self, porosity_path: str, cut_off_porosity: float) -> dict:
        """
        Функция для проверки того что Если в ячейке куба пористости, значение пористости меньше, чем cut-off, то куб ACTNUM = 0

            Required data:
                Porosity;
                ACTNUM;
                Cut-off_пористость|txt/xlsx|Керн|

            Args:
                porosity_path: str: путь к файлу
                cut_off_porosity: float: значение Cut-off_пористость
        """
        try:
            self.grid_model.add_prop(porosity_path, "PORO")
        except:
            return self.__generate_returns_dict(False, None, None)
        poro_value = self.grid_model.get_grid().get_prop_by_name("PORO")
        poro_value[np.isnan(poro_value)] = 0

        flag, wrong_data = self.__test_range_data(
            self.grid_model.get_grid().actnum_array,
            [lambda x: x[poro_value < cut_off_porosity] == 0]
        )

        if flag:
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
            return self.__generate_returns_dict(True, True, None)
        else:
            self.update_report(self.generate_report_text(
                f"Данные \n {wrong_data}  значение porosity < cut-off porosity, но ACTNUM == 1 ",
                Type_Status.NotPassed.value))
            return self.__generate_returns_dict(True, False, wrong_data)

    def test_swl_sw(self) -> dict:

        try:
            swl_value = self.__get_value_grid_prop(self.swl_file_path, False)

        except:
            return self.__generate_returns_dict(False, None, None)
        _, key = CubesTools().find_key(self.sw_file_path)
        flag, wrong_data = self.__test_value_conditions(
            key,
            [lambda x: x >= swl_value],
            sum
        )

        if flag:
            print("Тест пройден")
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
            return self.__generate_returns_dict(True, True, None)
        else:
            r_text = f"Данные <= SWL "
            print(f"Тест не пройден {r_text}")
            self.update_report(
                self.generate_report_text(
                    r_text,
                    Type_Status.NotPassed.value))
            return self.__generate_returns_dict(True, False, wrong_data)

    def __test_kern_data_dependence(
            self,
            data1,
            data2,
            kern_func):
        lit_data = self.__get_value_grid_prop(
            self.litatype_file_path,
            flag_d3=False
        )
        group_data_1, group_data_2 = CubesTools().get_cluster_dates(
            data1, data2, lit_data)

        mas_flag = []
        mas_wrong_data = []
        for cluster_key in group_data_1.keys():
            flag, wrong_data = kern_func(
                group_data_1[cluster_key],
                group_data_2[cluster_key],
                cluster_key)
            mas_flag.append(flag)
            mas_wrong_data.append(wrong_data)
            if flag:
                self.update_report(
                    self.generate_report_text(f"Скважина {cluster_key} прошла тест", Type_Status.Passed.value))
                print(f"Скважина {cluster_key} прошла тест")
            else:
                r_text = ""
                if wrong_data is None:
                    r_text += "Данные имеют None значение. "
                self.update_report(
                    self.generate_report_text(f"{r_text} Скважина {cluster_key} не прошла тест",
                                              Type_Status.Passed.value))
                print(f"Скважина {cluster_key} не прошла тест")

        if all(mas_flag):
            print("Тест пройден!!!")
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
        else:
            print("Тест не пройден!!!")
            self.update_report(self.generate_report_text(
                f"Зависимость по кубам не входит в кордиор вариации по керну",
                Type_Status.NotPassed.value))

    def test_kern_data_dependence_kpr_kp(self):
        self.__test_kern_data_dependence(
            data1=self.__get_value_grid_prop(
                self.open_porosity_file_path,
                flag_d3=False
            ),
            data2=self.__get_value_grid_prop(
                self.open_perm_x_file_path,
                flag_d3=False),
            kern_func=self.connector_kern.kern_test_dependence_kpr_kp,
        )

    def test_kern_data_dependence_kp_kgo(self):
        self.__test_kern_data_dependence(
            data1=self.__get_value_grid_prop(
                self.open_porosity_file_path,
                flag_d3=False
            ),
            data2=self.__get_value_grid_prop(
                self.sgl_file_path,
                flag_d3=False
            ),
            kern_func=self.connector_kern.kern_test_dependence_kp_kgo
        )

    def test_kern_data_dependence_kp_knmng(self):
        self.__test_kern_data_dependence(
            data1=self.__get_value_grid_prop(
                self.open_porosity_file_path,
                flag_d3=False
            ),
            data2=self.__get_value_grid_prop(
                self.sogcr_file_path_file_path_file_path,
                flag_d3=False
            ),
            kern_func=self.connector_kern.kern_test_dependence_kp_knmng
        )

    def test_kern_data_dependence_kp_kno(self):
        self.__test_kern_data_dependence(
            data1=self.__get_value_grid_prop(
                self.open_porosity_file_path,
                flag_d3=False
            ),
            data2=self.__get_value_grid_prop(
                self.sowcr_file_path_file_path,
                flag_d3=False
            ),
            kern_func=self.connector_kern.kern_test_dependence_kp_kno
        )

    def test_kern_data_dependence_quo_kp(self):
        self.__test_kern_data_dependence(
            data1=self.__get_value_grid_prop(
                self.open_porosity_file_path,
                flag_d3=False
            ),
            data2=self.__get_value_grid_prop(
                self.swl_file_path,
                flag_d3=False
            ),
            kern_func=self.connector_kern.kern_test_dependence_quo_kp
        )

    def test_kern_data_dependence_kpr_kgo(self):
        self.__test_kern_data_dependence(
            data1=self.__get_value_grid_prop(
                self.open_perm_x_file_path,
                flag_d3=False
            ),
            data2=self.__get_value_grid_prop(
                self.sgl_file_path,
                flag_d3=False
            ),
            kern_func=self.connector_kern.kern_test_dependence_kpr_kgo
        )

    def test_kern_data_dependence_kpr_knmng(self):
        self.__test_kern_data_dependence(
            data1=self.__get_value_grid_prop(
                self.open_perm_x_file_path,
                flag_d3=False
            ),
            data2=self.__get_value_grid_prop(
                self.sogcr_file_path,
                flag_d3=False
            ),
            kern_func=self.connector_kern.kern_test_dependence_kpr_knmng
        )

    def test_kern_data_dependence_kno_kpr(self):
        self.__test_kern_data_dependence(
            data1=self.__get_value_grid_prop(
                self.open_perm_x_file_path,
                flag_d3=False
            ),
            data2=self.__get_value_grid_prop(
                self.sowcr_file_path,
                flag_d3=False
            ),
            kern_func=self.connector_kern.kern_test_dependence_kno_kpr
        )

    def test_kern_data_dependence_kvo_kpr(self):
        self.__test_kern_data_dependence(
            data1=self.__get_value_grid_prop(
                self.open_perm_x_file_path,
                flag_d3=False
            ),
            data2=self.__get_value_grid_prop(
                self.swl_file_path,
                flag_d3=False
            ),
            kern_func=self.connector_kern.kern_test_dependence_kvo_kpr
        )

    def test_kern_data_dependence_kp_knmng(self):
        self.__test_kern_data_dependence(
            data1=self.__get_value_grid_prop(
                self.sogcr_file_path,
                flag_d3=False
            ),
            data2=self.__get_value_grid_prop(
                self.open_porosity_file_path,
                flag_d3=False
            ),
            kern_func=self.connector_kern.kern_test_dependence_kp_knmng
        )

    def test_kern_data_dependence_kpr_knmng(self):
        self.__test_kern_data_dependence(
            data1=self.__get_value_grid_prop(
                self.sogcr_file_path,
                flag_d3=False
            ),
            data2=self.__get_value_grid_prop(
                self.open_perm_x_file_path,
                flag_d3=False
            ),
            kern_func=self.connector_kern.kern_test_dependence_kpr_knmng
        )

    def test_kern_data_dependence_kno_kp(self):
        self.__test_kern_data_dependence(
            data1=self.__get_value_grid_prop(
                self.sowcr_file_path,
                flag_d3=False
            ),
            data2=self.__get_value_grid_prop(
                self.open_porosity_file_path,
                flag_d3=False
            ),
            kern_func=self.connector_kern.kern_test_dependence_kno_kp
        )

    def test_kern_data_dependence_kno_kpr(self):
        self.__test_kern_data_dependence(
            data1=self.__get_value_grid_prop(
                self.sowcr_file_path,
                flag_d3=False
            ),
            data2=self.__get_value_grid_prop(
                self.open_perm_x_file_path,
                flag_d3=False
            ),
            kern_func=self.connector_kern.kern_test_dependence_kno_kpr
        )

    def test_kern_data_dependence_quo_kp_2(self):
        self.__test_kern_data_dependence(
            data1=self.__get_value_grid_prop(
                self.swl_file_path,
                flag_d3=False
            ),
            data2=self.__get_value_grid_prop(
                self.open_porosity_file_path,
                flag_d3=False
            ),
            kern_func=self.connector_kern.kern_test_dependence_quo_kp_2
        )
