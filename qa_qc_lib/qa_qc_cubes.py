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
                 qa_qc_kern = None,
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
                 sw_file_path:str or None = None,
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
        self.actnum_file_path = actnum_file_path
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

        self.connector_kern = Connector_kern_cubes(qa_qc_kern, self)
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
            'Литотип':self.litatype_file_path,
        }

    def generate_report_text(self, text, status):
        status_dict = {0: 'Тест не пройден.',
                       1: 'Тест пройден успешно.',
                       2: 'Тест не был запущен.'}
        if status == 1 or status == 2:
            report_text = f"{self.ident}{status_dict[status]}\n{self.ident}{text}"
        else:
            report_text = f"{self.ident}{status_dict[status]}\n{self.ident}{text}\nФайл с кубом(не верные исходные данные) сохранен ->{self.save_wrong_data_path}/{inspect.stack()[1][3]}_WRONG_ACTNUM.GRDECL"
        return report_text
    def muc_np_arrays(self,np_array_list):
        result = np.ones(len(np_array_list[0]), 'bool')
        for n in np_array_list:
            result = n * result

        return result
    def __get_value_grid_prop(self, file_path: str, flag_d3: bool = True) -> np.array:
        key = CubesTools.find_key(file_path)
        self.grid_model.add_prop(file_path, key)
        prop = self.grid_model.get_grid().get_prop_by_name(key)
        value = self.grid_model.get_prop_value(prop, flag_d3)
        value[np.isnan(value)] = 0
        return value

    def __test_range_data(array, lambda_list: list[any], f) -> tuple[
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
        mask_array = (f([func(array) for func in lambda_list])).astype(dtype=bool)

        if all(mask_array):
            return True, None
        else:
            return False, mask_array == False

    def __test_value_conditions(self,save_wrong_data_path: str, file_path: str, prop_name: str, lambda_list: list[any], f) -> tuple[
        bool
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
            lambda_list,
            f)

        if flag:
            return True
        else:
            CubesTools.generate_wrong_actnum(wrong_data, self.save_wrong_data_path)
            return False

    """
    Тесты первого порядка
    """

    def test_open_porosity(self):
        """
        Функция для проверки открытой пористости

            Required data:
                Porosity;
            Args:
                file_path: str: путь к файлу
                prop_name: str: ключ
        """
        flag = self.__test_value_conditions(
            self.open_porosity_file_path,
            CubesTools.find_key(self.open_porosity_file_path),
            [lambda x: x > 0, lambda x: x <= 0.476],
            self.muc_np_arrays
        )

        if flag:
            self.generate_report_text("", Type_Status.Passed.value)
        else:
            self.generate_report_text(f"Данные лежат не в интервале от 0 до 47,6",
                                      Type_Status.NotPassed)

    def test_permeability(self, name_type_data: str):
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
        if name_type_data in self.help_dict_type_data:
            self.update_report(self.generate_report_text("Названия типа данных не существует", Type_Status.NotRunning.value))
            return

        flag = self.__test_value_conditions(
            self.help_dict_type_data[name_type_data],
            CubesTools.find_key(self.help_dict_type_data[name_type_data]),
            [lambda x: x >= 0],
            sum
        )

        if flag:
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
        else:
            self.update_report(self.generate_report_text(
                f"Данные < 0",
                Type_Status.NotPassed))

    def test_range_data(self, name_type_data: str):
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
        if name_type_data in self.help_dict_type_data:
            self.update_report(self.generate_report_text("Названия типа данных не существует", Type_Status.NotRunning.value))
            return

        flag = self.__test_value_conditions(
            self.help_dict_type_data[name_type_data],
            CubesTools.find_key(self.help_dict_type_data[name_type_data]),
            [lambda x: x >= 0, lambda x: x <= 1],
            self.muc_np_arrays
        )

        if flag:
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
        else:
            self.update_report(self.generate_report_text(
                f"Данные лежат не в интервале от 0 до 1 ",
                Type_Status.NotPassed))

    def test_range_integer_data(self):
        """
        Функция для проверки данных на x == 0 || x == 1

            Required data:
                ACTNUM;

            Args:
                file_path: str: путь к файлу
                prop_name: str: ключ
        """
        flag = self.__test_value_conditions(
            self.actnum_file_path,
            CubesTools.find_key(self.actnum_file_path),
            [lambda x: x == 0, lambda x: x == 1],
            sum
        )

        if flag:
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
        else:
            self.update_report(self.generate_report_text(
                f"Данные не равняются 0 или 1",
                Type_Status.NotPassed))

    def test_integer_data(self, file_path: str, prop_name: str):
        """
        Функция для проверки данных на x целое число

            Required data:
                Литотип;

            Args:
                file_path: str: путь к файлу
                prop_name: str: ключ
        """
        flag = self.__test_value_conditions(
            file_path,
            prop_name
            [lambda x: x % 1 == 0]
        )

        if flag:
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
        else:
            self.update_report(self.generate_report_text(
                f"Данные не целочисленные",
                Type_Status.NotPassed))

    def test_bulk(self):
        """
        Функция для проверки данных геометрического объема grid-a, должен быть не отрицательным
        """
        data = self.grid_model.get_grid().get_bulk_volume(asmasked=False)
        data[np.isnan(data)] = 0

        flag, wrong_data = self.__test_range_data(
            data,
            [lambda x: x >= 0],
            sum
        )

        if flag:
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
        else:
            CubesTools.generate_wrong_actnum(wrong_data, self.save_wrong_data_path)
            self.update_report(self.generate_report_text(
                f"Данные имеют отрицательный объем",
                Type_Status.NotPassed))

    """
    Тесты второго порядка
    """

    def test_sum_cubes(self, list_name_type_file: list[str]):
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

            self.grid_model.add_prop(
                self.help_dict_type_data[name_type_file],
                CubesTools.find_key(self.help_dict_type_data[name_type_file])
            )

            data_mas.append(
                self.grid_model.get_prop_value(
                    self.grid_model.get_grid().get_prop_by_name(
                        CubesTools.find_key(
                            self.help_dict_type_data[name_type_file])
                    )
                )
            )

        flag, wrong_data = self.__test_range_data(
            data_mas,
            [lambda x: sum(x) == 1]
        )

        if flag:
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
        else:
            CubesTools.generate_wrong_actnum(wrong_data, self.save_wrong_data_path)
            self.update_report(self.generate_report_text(
                f"Cумма кубов != 1",
                Type_Status.NotPassed))

    def test_affiliation_sqcr(self, sgcr_path: str, sgl_path: str):
        """
        Функция для проверки того что SGCR Є [SGL:1]

            Required data:
                SGCR;
                SGL;

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
                SGCR;
                SGL;

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
                Porosity;
                ACTNUM;
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
    def test_kern_data_dependence_kpr_kp(self):
        poro_group_data, permX_group_data = CubesTools.get_cluster_dates(
            self.open_porosity_file_path,
            self.open_perm_x_file_path,
            self.litatype_file_path,
        )

        mas_flag = []
        mas_wrong_data = []

        for cluster_key in poro_group_data.keys():
            flag, wrong_data = self.connector_kern.kern_test_dependence_kpr_kp(poro_group_data[cluster_key], permX_group_data[cluster_key])
            mas_flag.append(flag)
            mas_wrong_data.append(wrong_data)

        if all(mas_flag):
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
        else:
            CubesTools.generate_wrong_actnum(
                self.muc_np_arrays(mas_wrong_data),
                self.save_wrong_data_path)

            self.update_report(self.generate_report_text(
                f"Зависимость по кубам не входит в кордиор вариации по керну",
                Type_Status.NotPassed))

    def test_kern_data_dependence_kp_sgl(self):
        poro_group_data, permX_group_data = CubesTools.get_cluster_dates(
            self.open_porosity_file_path,
            self.open_perm_x_file_path,
            self.litatype_file_path,
        )
        for index in range(len(poro_group_data)):
            flag, wrong_data = self.connector_kern.kern_test_dependence_kpr_kp(poro_group_data, permX_group_data)
            if flag:
                self.update_report(self.generate_report_text("", Type_Status.Passed.value))
            else:
                CubesTools.generate_wrong_actnum(wrong_data, self.save_wrong_data_path)
                self.update_report(self.generate_report_text(
                    f"Зависимость по кубам не входит в кордиор вариации по керну",
                    Type_Status.NotPassed))

# self.generate_test_report(file_name=self.__class__.__name__, file_path="../../report", data_name=file_path)


#QA_QC_cubes("../../data/grdecl_data/GRID.GRDECL").test_open_porosity("../../data/grdecl_data/input/Poro.GRDECL.grdecl")
