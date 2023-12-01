import numpy as np
from sklearn.linear_model import LinearRegression
from qa_qc_lib.tools.cubes_tools import CubesTools

from matplotlib import pyplot as plt
from qa_qc_lib.tests.base_test import QA_QC_main


class Connector_kern_cubes(QA_QC_main):

    def __init__(
            self,
            qa_qc_kern,
            qa_qc_cubes):
        QA_QC_main.__init__(self)
        self.QA_QC_kern = qa_qc_kern
        self.QA_QC_cubes = qa_qc_cubes

    def linear_regressor(self, data_array_1: np.array, data_array_2: np.array):
        linear_regressor = LinearRegression().fit(data_array_1.reshape(-1, 1),
                                                  data_array_2.reshape(-1, 1))  # perform linear regression

        # The coefficients of linear gerression
        k = linear_regressor.coef_
        b = linear_regressor.intercept_

        return (k, b)

    def sigma_counter(self, flat_data: np.array, how_many_sigmas=1):
        return (
            flat_data.mean() - how_many_sigmas * flat_data.std(), flat_data.mean() + how_many_sigmas * flat_data.std())

    def borders_initializer(self, data_array_1: np.array,
                            data_array_2: np.array,
                            outer_limit=3):
        X_max = data_array_1.max()
        X_min = data_array_1.min()

        k, b = self.linear_regressor(data_array_1, data_array_2)

        flat_array = data_array_2 - (k * data_array_1 + b)
        flat_array[flat_array < 0] = 0
        sigma_min, sigma_max = self.sigma_counter(flat_array, outer_limit)

        gamma_min = k * X_min + b + sigma_min
        gamma_max = k * X_min + b + sigma_max

        beta_min = k * X_max + b + sigma_min
        beta_max = k * X_max + b + sigma_max

        x_out_down, y_out_down = [X_min, X_max], [gamma_min.item(), beta_min.item()]
        x_out_up, y_out_up = [X_min, X_max], [gamma_max.item(), beta_max.item()]

        return (flat_array[0],
                [
                    x_out_down, y_out_down
                ],
                [
                    x_out_up, y_out_up
                ])

    def is_point_line(self, points_x, points_y, test_point, _lambda) -> bool:
        x1, x2 = points_x
        y1, y2 = points_y
        x3, y3 = test_point

        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        expected_y = m * x3 + b
        return _lambda(y3, expected_y)

    def draw_plot(self, name, points_true, points_false, points_core, line_up, line_down, x_l, y_l):
        plt.title(f"Cопоставление измеренного и модельного распределений")
        plt.scatter(x=points_false[0], y=points_false[1], s=3, marker='o', color='r', label="cubes_points_false")
        plt.scatter(x=points_true[0], y=points_true[1], s=3, marker='o', color='g', label="cubes_points_true")
        plt.scatter(x=points_core[0], y=points_core[1], s=3, marker='o', color='b', label="core_data_points")
        plt.plot(line_down[0], line_down[1], label='mean + 3σ, mean - 3σ', color='b')
        plt.plot(line_up[0], line_up[1], color='b')
        plt.xlabel(x_l)
        plt.ylabel(y_l)
        plt.legend()
        plt.show()
        plt.close()

    def check_data_point(self, data_x_1, data_y_1, data_x_2, data_y_2, key, x_l, y_l):
        _, hallway_kern_down, hallway_kern_up = self.borders_initializer(data_x_1, data_y_1)
        _float, _, _ = self.borders_initializer(data_x_2, data_y_2)
        result_array = []
        for index in range(len(data_x_2)):
            result_array.append(
                self.is_point_line(
                    hallway_kern_down[0],
                    hallway_kern_down[1],
                    (data_x_2[index], data_y_2[index]),
                    lambda x, y: x >= y)
                and
                self.is_point_line(
                    hallway_kern_up[0],
                    hallway_kern_up[1],
                    (data_x_2[index], data_y_2[index]),
                    lambda x, y: x <= y))

        result = np.array(result_array)

        self.draw_plot(key, [data_x_2[result], data_y_2[result]],
                       [data_x_2[result == False], data_y_2[result == False]], [data_x_1, data_y_1], hallway_kern_up,
                       hallway_kern_down, x_l, y_l)
        if all(result):
            return True, None
        else:
            return False, result == False

    def __abstract_kern_data_dependence(
            self,
            data1,
            data2,
            kern_func) -> dict:
        lit_data = self.__get_value_grid_prop(
            self.litatype_file_path,
            flag_d3=False
        )
        group_data_1, group_data_2 = CubesTools().get_cluster_dates(
            data1, data2, lit_data)

        mas_flag = []
        mas_wrong_data = []
        wrong_data = None
        for cluster_key in group_data_1.keys():
            flag, wrong_data = kern_func(
                group_data_1[cluster_key],
                group_data_2[cluster_key],
                cluster_key)
            mas_flag.append(flag)
            mas_wrong_data.append(wrong_data)
            if flag:
                self.update_report(
                    self.generate_report_text(f"Скважина {cluster_key} прошла тест", 1))
            else:
                r_text = ""
                if wrong_data is None:
                    r_text += "Данные имеют None значение. "
                self.update_report(
                    self.generate_report_text(f"{r_text} Скважина {cluster_key} не прошла тест",
                                              0))

        if all(mas_flag):
            self.update_report(self.generate_report_text("", 1))
            return self.QA_QC_cubes.__generate_returns_dict(True, True, None)
        else:
            self.update_report(self.generate_report_text(
                f"Зависимость по кубам не входит в кордиор вариации по керну",
                0))
            return self.QA_QC_cubes.__generate_returns_dict(True, False, wrong_data)

    def __kern_test_dependence(self,
                               cube_group_data_1: np.array,
                               cube_group_data_2: np.array,
                               kern_group_data_1: np.array,
                               kern_group_data_2: np.array,
                               cluster_key: str,
                               x_l="data1",
                               y_l="data2") -> (bool, np.array or None):

        kern_group_data_1[np.isnan(kern_group_data_1)] = 0
        kern_group_data_2[np.isnan(kern_group_data_2)] = 0
        cluster1 = None
        cluster2 = None

        if cluster_key != "lit_none":
            lithotype = self.QA_QC_kern.lithotype
            cluster1, cluster2 = CubesTools().get_cluster_dates(kern_group_data_1, kern_group_data_2, lithotype)
        else:
            cluster1 = {"lit_none": kern_group_data_1}
            cluster2 = {"lit_none": kern_group_data_2}
        if cluster_key in cluster1:
            return self.check_data_point(
                cluster1[cluster_key],
                cluster2[cluster_key],
                cube_group_data_1,
                cube_group_data_2,
                cluster_key,
                x_l,
                y_l)
        return False, "Not Key"

    def __kern_test_dependence_quo_kp(
            self,
            poro_group_data: np.array,
            swl_group_data: np.array,
            cluster_key) -> (bool, np.array or None):
        return self.__kern_test_dependence(
            poro_group_data,
            swl_group_data,
            self.QA_QC_kern.porosity_open,
            self.QA_QC_kern.sw_residual,
            cluster_key,
            "Пористость,  v/v",
            "Проницаемость, мД")

    def test_kern_data_dependence_quo_kp(self):
        """
        Проверка соответствия модельных данных (в кубах) данным, определенным по керну.
        Базовые петрофизические зависимости задают диапазон неопределенности. Все модельные данные должны быть получены в измеренных диапазонах.


            Required data:
                SWL|GRDECL|ПЕТРОФИЗИКА|
                Кво|txt(xlsx)|Керн|,Кп_абс|txt(xlsx)|
                Керн|/Кп_откр|txt(xlsx)|
                Керн|,Литотип|txt(xlsx)|
                Керн|,Фации|txt(xlsx)|
                Керн|,Porosity|GRDECL|
                ПЕТРОФИЗИКА|,Литотип|GRDECL|
                ПЕТРОФИЗИКА|,Фации|GRDECL|
                ПЕТРОФИЗИКА|,Кп_абс|las|
                ПЕТРОФИЗИКА|,Кп_октр|las|
                ПЕТРОФИЗИКА|,Кво(SWL)|las|
                ПЕТРОФИЗИКА|,Литотип|las|
                ПЕТРОФИЗИКА|,Фации|las|
                ПЕТРОФИЗИКА|,GRID_модели|GRDECL|Сейсмика|

            Returns:
                 dict: Словарь, specification cловарь где ,wrong_data - список ячеек куба которые не прошли тестирование
        """
        self.__abstract_kern_data_dependence(
            data1=self.QA_QC_cubes.__get_value_grid_prop(
                self.QA_QC_cubes.open_porosity_file_path,
                flag_d3=False
            ),
            data2=self.QA_QC_cubes.__get_value_grid_prop(
                self.QA_QC_cubes.swl_file_path,
                flag_d3=False
            ),
            kern_func=self.__kern_test_dependence_quo_kp
        )

    def __kern_test_dependence_kpr_kp(
            self,
            poro_group_data: np.array,
            permX_group_data: np.array,
            cluster_key) -> (bool, np.array or None):
        return self.__kern_test_dependence(
            poro_group_data,
            permX_group_data,
            self.QA_QC_kern.porosity_open,
            self.QA_QC_kern.kpr,
            cluster_key,
            "Пористость,  v/v",
            "Проницаемость, мД")

    def test_kern_data_dependence_kpr_kp(self):
        """
        Проверка соответствия модельных данных (в кубах) данным, определенным по керну.
        Базовые петрофизические зависимости задают диапазон неопределенности. Все модельные данные должны быть получены в измеренных диапазонах.
        Кубы должы соответствовать изначальной петрофизической зависимости.


            Required data:
                PermX|GRDECL|ПЕТРОФИЗИКА|
                Кп_откр|txt(xlsx)|Керн|
                Кп_абс|txt(xlsx)|Керн|
                Кпр_абс|txt(xlsx)|Керн|
                Литотип|txt(xlsx)|Керн|
                Фации|txt(xlsx)|Керн|
                Porosity|GRDECL|ПЕТРОФИЗИКА|
                Литотип|GRDECL|ПЕТРОФИЗИКА|
                Фации|GRDECL|ПЕТРОФИЗИКА|
                Кпр_абс|las|ПЕТРОФИЗИКА|
                Кп_октр|las|ПЕТРОФИЗИКА|
                Кп_абс|las|ПЕТРОФИЗИКА|
                Литотип|las|ПЕТРОФИЗИКА|
                Фации|las|ПЕТРОФИЗИКА|

            Returns:
                 dict: Словарь, specification cловарь где ,wrong_data - список ячеек куба которые не прошли тестирование
        """
        self.__abstract_kern_data_dependence(
            data1=self.QA_QC_cubes.__get_value_grid_prop(
                self.QA_QC_cubes.open_porosity_file_path,
                flag_d3=False
            ),
            data2=self.QA_QC_cubes.__get_value_grid_prop(
                self.QA_QC_cubes.open_perm_x_file_path,
                flag_d3=False),
            kern_func=self.__kern_test_dependence_kpr_kp,
        )