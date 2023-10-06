import copy
import inspect

from qa_qc_lib.data_reader import QA_QC_grdecl_parser
from qa_qc_lib.qa_qc_main import QA_QC_main, Type_Status
from qa_qc_lib.qa_qc_tools.cubes_tools import CubesTools
from qa_qc_lib.qa_qc_connectors.connector_kern_cubes import Connector_kern_cubes
import numpy as np

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

        if qa_qc_kern is not None:
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
            'Литотип': self.litatype_file_path,
        }

        self.actnum  = self.grid_model.get_grid().get_actnum().get_npvalues3d()

        self.grid_head = CubesTools().find_head(f"{directory_path}/{grid_name}_ACTNUM.GRDECL")

        attributes = locals()
        for key in attributes.keys():
            if 'file' in key and attributes[key] is not None:
                flag, prop_name = CubesTools().find_key(attributes[key])
                if flag:
                    print(attributes[key], "->", prop_name)
                    self.grid_model.add_prop(attributes[key], prop_name)

    def generate_report_text(self, text, status):
        status_dict = {0: 'Тест не пройден.',
                       1: 'Тест пройден успешно.',
                       2: 'Тест не был запущен.'}
        if status == 1 or status == 2:
            report_text = f"{self.ident}{status_dict[status]}\n{self.ident}{text}"
        else:
            report_text = f"{self.ident}{status_dict[status]}\n{self.ident}{text}\nФайл с кубом(не верные исходные данные) сохранен ->{self.save_wrong_data_path}/{inspect.stack()[1][3]}_WRONG_ACTNUM.GRDECL"
        return report_text

    def muc_np_arrays(self, np_array_list):
        result = np.ones_like(len(np_array_list[0]), 'bool')
        for n in np_array_list:
            result = n * result

        return result

    def __get_value_grid_prop(self, file_path: str) -> np.array:
        _, key = CubesTools().find_key(file_path)
        prop = self.grid_model.get_grid().get_prop_by_name(key)
        value = self.grid_model.get_prop_value(prop)
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
            return False, copy_actnum

    def __test_value_conditions(self, prop_name: str, lambda_list: list[any], f) -> tuple[
        bool
    ]:
        """
        Функция для парса данных и отправки их на проверку

            Args:
                prop_name: str: ключ
                lambda_list: list[any]: список с вырожениями для проверки

            Returns:
                bool: результат тестирования
                str or None: страка с результатом
        """

        flag, wrong_actnum = self.__test_range_data(
            self.grid_model.get_prop_value(
                self.grid_model.get_grid().get_prop_by_name(prop_name)
            ),
            lambda_list=lambda_list,
            f=f)

        if flag:
            return True
        else:
            self.grid_model.generate_wrong_actnum(wrong_actnum, self.grid_head, self.save_wrong_data_path, inspect.stack()[1][3])
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
        _, key = CubesTools().find_key(self.open_porosity_file_path)
        flag = self.__test_value_conditions(
            prop_name=key,
            lambda_list=[lambda x: x >= 0, lambda x: x <= 0.476],
            f=self.muc_np_arrays
        )

        if flag:
            print("Тест пройден")
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
        else:
            r_text = f"Данные лежат не в интервале от 0 до 0.476"
            print(f"Тест не пройден {r_text}")
            self.update_report(self.generate_report_text(f"Данные лежат не в интервале от 0 до 47,6",
                                                         Type_Status.NotPassed.value))

    def __abstract_test_permeability(self, file_path):
        """
        Функция для проверки данных на x >= 0

            Required data:
                PermX;
                PermY;
                PermZ;
                J-function;
            Args:
                file_path: str: путь к файлу
        """
        _, key = CubesTools().find_key(file_path)
        flag = self.__test_value_conditions(
            key,
            [lambda x: x >= 0],
            sum
        )

        if flag:
            print("Тест пройден")
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
        else:
            r_text = f"Данные < 0"
            print(f"Тест не пройден {r_text}")
            self.update_report(self.generate_report_text(
                r_text,
                Type_Status.NotPassed.value))

    def test_permeability_permX(self):
        """
                   Required data:
                       PermX;
               """
        self.__abstract_test_permeability(self.open_perm_x_file_path)

    def test_permeability_permY(self):
        """
                           Required data:
                               PermY;
                       """
        self.__abstract_test_permeability(self.open_perm_y_file_path)

    def test_permeability_permZ(self):
        """
                           Required data:
                               PermZ;
                       """
        self.__abstract_test_permeability(self.open_perm_z_file_path)
    def __abstract_test_range_data(self, file_path):
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
        """
        _, key = CubesTools().find_key(file_path)

        flag = self.__test_value_conditions(
            key,
            [lambda x: x >= 0, lambda x: x <= 1],
            self.muc_np_arrays
        )

        if flag:
            print("Тест пройден")
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
        else:
            r_text = f"Данные лежат не в интервале от 0 до 1"
            print(f"Тест не пройден {r_text}")
            self.update_report(self.generate_report_text(
                r_text,
                Type_Status.NotPassed.value))

    def test_range_data_sgsr(self):
        """
                   Required data:
                       SGCR;
        """
        self.__abstract_test_range_data(self.sgcr_file_path)

    def test_range_data_sgl(self):
        """
                           Required data:
                               SGL;

                """
        self.__abstract_test_range_data(self.sgcr_file_path)

    def test_range_data_swatinit(self):
        """
                           Required data:
                               SWATINIT;

                """
        self.__abstract_test_range_data(self.sw_file_path)

    def test_range_data_sgu(self):
        """
                           Required data:
                               SGU;
                """
        self.__abstract_test_range_data(self.sgu_file_path)

    def test_range_data_swl(self):
        """
                           Required data:
                               SWL;
                """
        self.__abstract_test_range_data(self.swl_file_path)

    def test_range_data_swcr(self):
        """
                           Required data:
                               SWCR;
                """
        self.__abstract_test_range_data(self.swcr_file_path)

    def test_range_data_swu(self):
        """
                           Required data:
                               SWU;
                """
        self.__abstract_test_range_data(self.swu_file_path)

    def test_range_data_ntg(self):
        """
                           Required data:
                               NTG;
                """
        self.__abstract_test_range_data(self.ntg_file_path)

    def test_range_data_so(self):
        """
                           Required data:
                               So;
                """
        self.__abstract_test_range_data(self.so_file_path)

    def test_range_data_sg(self):
        """
                           Required data:
                               Sg;
                """
        self.__abstract_test_range_data(self.sg_file_path)

    def test_range_data_sogcr(self):
        """
                           Required data:
                               SOGCR;
                """
        self.__abstract_test_range_data(self.sgl_file_path)
    def test_right_actnum(self):
        """
        Функция для проверки данных на x == 0 || x == 1

            Required data:
                ACTNUM;

            Args:
                file_path: str: путь к файлу
        """
        actnum_array = self.grid_model.get_grid().get_actnum().values

        flag, wrong_grid = self.__test_range_data(
            actnum_array,
            [lambda x: x == 0, lambda x: x == 1],
            sum
        )

        if flag:
            print("Тест пройден")
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
        else:
            self.grid_model.generate_wrong_actnum(wrong_grid, self.grid_head, self.save_wrong_data_path, "test_right_actnum")
            r_text = f"Данные не равняются 0 или 1"
            print(f"Тест не пройден {r_text}")
            self.update_report(self.generate_report_text(
                f"Данные не равняются 0 или 1",
                Type_Status.NotPassed.value))

    def test_litatype(self):
        """
        Функция для проверки данных на x целое число

            Required data:
                Литотип;

            Args:
                file_path: str: путь к файлу
        """
        _, key = CubesTools().find_key(self.litatype_file_path)
        flag = self.__test_value_conditions(
            key,
            [lambda x: x % 1 == 0],
            sum
        )

        if flag:
            print("Тест пройден")
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
        else:
            r_text = f"Данные не целочисленные"
            print(f"Тест не пройден {r_text}")
            self.update_report(self.generate_report_text(
                r_text,
                Type_Status.NotPassed.value))

    def test_bulk(self):
        """
        Функция для проверки данных геометрического объема grid-a, должен быть не отрицательным
        """
        data = self.grid_model.get_grid().get_bulk_volume(asmasked=False).get_npvalues3d()

        flag, wrong_grid = self.__test_range_data(
            data,
            [lambda x: x >= 0],
            sum
        )

        if flag:
            print("Тест пройден")
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
        else:
            r_text = f"Данные имеют отрицательный объем"
            print(f"Тест не пройден {r_text}")
            self.grid_model.generate_wrong_actnum(wrong_grid, self.grid_head, self.save_wrong_data_path, "test_bulk")
            self.update_report(self.generate_report_text(
                r_text,
                Type_Status.NotPassed.value))

    """
    Тесты второго порядка
    """

    def test_sum_cubes(self):
        """
        Функция для проверки того что сумма кубов = 1

            Required data:
                SWATINIT;
                Sg;
                So;
        """
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
            [lambda x: x == 1],
            sum
        )

        if flag:
            print("Тест пройден")
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
        else:
            r_text = f"Cумма кубов != 1"
            print(f"Тест не пройден {r_text}")
            self.grid_model.generate_wrong_actnum(wrong_data, self.grid_head, self.save_wrong_data_path, "test_sum_cubes")
            self.update_report(self.generate_report_text(
                r_text,
                Type_Status.NotPassed.value))

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

        sgcr_value = self.__get_value_grid_prop(self.sgcr_file_path)
        sgl_value = self.__get_value_grid_prop(self.sgl_file_path)

        flag, wrong_data = self.__test_range_data(
            sgcr_value,
            [lambda x: x >= sgl_value[self.actnum == 1], lambda x: x <= 1],
            self.muc_np_arrays
        )

        if flag:
            print("Тест пройден")
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
        else:
            r_text = f"Данные  ∉ [SGL:1] "
            print(f"Тест не пройден {r_text}")
            self.grid_model.generate_wrong_actnum(wrong_data, self.grid_head, self.save_wrong_data_path, "test_affiliation_sqcr")
            self.update_report(self.generate_report_text(
                r_text,
                Type_Status.NotPassed.value))

    def test_affiliation_swcr(self):
        """
        Функция для проверки того что SWCR ≥ SWL

            Required data:
                SGCR;
                SGL;
        """
        swcr_value = self.__get_value_grid_prop(self.swcr_file_path)

        swl_value = self.__get_value_grid_prop(self.swl_file_path)

        flag, wrong_data = self.__test_range_data(
            swcr_value,
            [lambda x: x >= swl_value[self.actnum == 1]],
            sum
        )

        if flag:
            print("Тест пройден")
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
        else:
            r_text = f"Данные < SWL "
            print(f"Тест не пройден {r_text}")
            self.grid_model.generate_wrong_actnum(wrong_data, self.grid_head, self.save_wrong_data_path, "test_affiliation_swcr")
            self.update_report(self.generate_report_text(
                r_text,
                Type_Status.NotPassed.value))

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
                Type_Status.NotPassed.value))

    def test_swl_sw(self):
        """
                Проверка SWATINIT >= SWL

                    Required data:
                        SWATINIT
                        SWL

                    Args:
                        porosity_path: str: путь к файлу
                        cut_off_porosity: float: значение Cut-off_пористость
                """
        swl_value = self.__get_value_grid_prop(self.swl_file_path)
        _, key = CubesTools().find_key(self.sw_file_path)
        flag = self.__test_value_conditions(
            key,
            [lambda x: x >= swl_value[self.actnum == 1]],
            sum
        )

        if flag:
            print("Тест пройден")
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
        else:
            r_text = f"Данные <= SWL "
            print(f"Тест не пройден {r_text}")
            self.update_report(
                self.generate_report_text(
                    r_text,
                    Type_Status.NotPassed.value))

    def __test_kern_data_dependence(
            self,
            data1,
            data2,
            kern_func):

        data1 = data1[self.actnum == 1]
        data2 = data2[self.actnum == 1]
        group_data_1 = None
        group_data_2 = None
        if self.litatype_file_path is not None:
            lit_data = CubesTools().conver_n3d_to_n1d(self.__get_value_grid_prop(
                self.litatype_file_path,
            ))

            group_data_1, group_data_2 = CubesTools().get_cluster_dates(
                data1, data2, lit_data)
        else:
            group_data_1 = {"lit_none": data1}
            group_data_2 = {"lit_none": data2}

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
                    self.generate_report_text(f"{r_text} Скважина {cluster_key} не прошла тест", Type_Status.Passed.value))
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
                self.open_porosity_file_path
            ),
            data2=self.__get_value_grid_prop(
                self.open_perm_x_file_path),
            kern_func=self.connector_kern.kern_test_dependence_kpr_kp,
        )

    def test_kern_data_dependence_kp_kgo(self):
        self.__test_kern_data_dependence(
            data1=self.__get_value_grid_prop(
                self.open_porosity_file_path
            ),
            data2=self.__get_value_grid_prop(
                self.sgl_file_path
            ),
            kern_func=self.connector_kern.kern_test_dependence_kp_kgo
        )

    def test_kern_data_dependence_kp_knmng(self):
        self.__test_kern_data_dependence(
            data1=self.__get_value_grid_prop(
                self.open_porosity_file_path
            ),
            data2=self.__get_value_grid_prop(
                self.sogcr_file_path_file_path_file_path
            ),
            kern_func=self.connector_kern.kern_test_dependence_kp_knmng
        )

    def test_kern_data_dependence_kp_kno(self):
        self.__test_kern_data_dependence(
            data1=self.__get_value_grid_prop(
                self.open_porosity_file_path
            ),
            data2=self.__get_value_grid_prop(
                self.sowcr_file_path_file_path
            ),
            kern_func=self.connector_kern.kern_test_dependence_kp_kno
        )

    def test_kern_data_dependence_quo_kp(self):
        self.__test_kern_data_dependence(
            data1=self.__get_value_grid_prop(
                self.open_porosity_file_path
            ),
            data2=self.__get_value_grid_prop(
                self.swl_file_path
            ),
            kern_func=self.connector_kern.kern_test_dependence_quo_kp
        )

    def test_kern_data_dependence_kpr_kgo(self):
        self.__test_kern_data_dependence(
            data1=self.__get_value_grid_prop(
                self.open_perm_x_file_path
            ),
            data2=self.__get_value_grid_prop(
                self.sgl_file_path
            ),
            kern_func=self.connector_kern.kern_test_dependence_kpr_kgo
        )

    def test_kern_data_dependence_kpr_knmng(self):
        self.__test_kern_data_dependence(
            data1=self.__get_value_grid_prop(
                self.open_perm_x_file_path
            ),
            data2=self.__get_value_grid_prop(
                self.sogcr_file_path
            ),
            kern_func=self.connector_kern.kern_test_dependence_kpr_knmng
        )

    def test_kern_data_dependence_kno_kpr(self):
        self.__test_kern_data_dependence(
            data1=self.__get_value_grid_prop(
                self.open_perm_x_file_path
            ),
            data2=self.__get_value_grid_prop(
                self.sowcr_file_path
            ),
            kern_func=self.connector_kern.kern_test_dependence_kno_kpr
        )

    def test_kern_data_dependence_kvo_kpr(self):
        self.__test_kern_data_dependence(
            data1=self.__get_value_grid_prop(
                self.open_perm_x_file_path
            ),
            data2=self.__get_value_grid_prop(
                self.swl_file_path
            ),
            kern_func=self.connector_kern.kern_test_dependence_kvo_kpr
        )

    def test_kern_data_dependence_kp_knmng(self):
        self.__test_kern_data_dependence(
            data1=self.__get_value_grid_prop(
                self.sogcr_file_path
            ),
            data2=self.__get_value_grid_prop(
                self.open_porosity_file_path
            ),
            kern_func=self.connector_kern.kern_test_dependence_kp_knmng
        )

    def test_kern_data_dependence_kpr_knmng(self):
        self.__test_kern_data_dependence(
            data1=self.__get_value_grid_prop(
                self.sogcr_file_path
            ),
            data2=self.__get_value_grid_prop(
                self.open_perm_x_file_path
            ),
            kern_func=self.connector_kern.kern_test_dependence_kpr_knmng
        )

    def test_kern_data_dependence_kno_kp(self):
        self.__test_kern_data_dependence(
            data1=self.__get_value_grid_prop(
                self.sowcr_file_path
            ),
            data2=self.__get_value_grid_prop(
                self.open_porosity_file_path
            ),
            kern_func=self.connector_kern.kern_test_dependence_kno_kp
        )

    def test_kern_data_dependence_kno_kpr(self):
        self.__test_kern_data_dependence(
            data1=self.__get_value_grid_prop(
                self.sowcr_file_path
            ),
            data2=self.__get_value_grid_prop(
                self.open_perm_x_file_path
            ),
            kern_func=self.connector_kern.kern_test_dependence_kno_kpr
        )

    def test_kern_data_dependence_quo_kp_2(self):
        self.__test_kern_data_dependence(
            data1=self.__get_value_grid_prop(
                self.swl_file_path
            ),
            data2=self.__get_value_grid_prop(
                self.open_porosity_file_path
            ),
            kern_func=self.connector_kern.kern_test_dependence_quo_kp_2
        )

# self.generate_test_report(file_name=self.__class__.__name__, file_path="../../report", data_name=file_path)


# QA_QC_cubes("../../data/grdecl_data/GRID.GRDECL").test_open_porosity("../../data/grdecl_data/input/Poro.GRDECL.grdecl")
