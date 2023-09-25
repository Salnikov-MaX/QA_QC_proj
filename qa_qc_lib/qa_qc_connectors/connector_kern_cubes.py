import numpy as np
from sklearn.linear_model import LinearRegression
from qa_qc_lib.qa_qc_tools.cubes_tools import CubesTools

def linear_regressor(data_array_1: np.array, data_array_2: np.array):
    linear_regressor = LinearRegression().fit(data_array_1.reshape(-1, 1),
                                              data_array_2.reshape(-1, 1))  # perform linear regression

    # The coefficients of linear gerression
    k = linear_regressor.coef_
    b = linear_regressor.intercept_

    return (k, b)


def sigma_counter(flat_data: np.array, how_many_sigmas=1):
    return (flat_data.mean() - how_many_sigmas * flat_data.std(), flat_data.mean() + how_many_sigmas * flat_data.std())


def borders_initializer(data_array_1: np.array,
                        data_array_2: np.array,
                        outer_limit=3):
    X_max = data_array_1.max()
    X_min = data_array_1.min()

    k, b = linear_regressor(data_array_1, data_array_2)

    flat_array = data_array_2 - (k * data_array_1 + b)

    sigma_min, sigma_max = sigma_counter(flat_array, outer_limit)

    gamma_min = k * X_min + b + sigma_min
    gamma_max = k * X_min + b + sigma_max

    beta_min = k * X_max + b + sigma_min
    beta_max = k * X_max + b + sigma_max

    x_out_down, y_out_down = [X_min, X_max], [gamma_min.item(), beta_min.item()]
    x_out_up, y_out_up = [X_min, X_max], [gamma_max.item(), beta_max.item()]

    return (flat_array[0],
            [
                (x_out_down[0], y_out_down[0]), (x_out_down[1], x_out_down[1])
            ],
            [
                (x_out_up[0], y_out_up[0]), (x_out_up[1], x_out_up[1])
            ])


def is_point_line(point1, point2, test_point, _lambda) -> bool:
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = test_point

    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1

    expected_y = m * x3 + b
    return _lambda(y3, expected_y)

def check_data_point(data_x_1,data_y_1,data_x_2,data_y_2):
    _, hallway_kern_down, hallway_kern_up = borders_initializer(data_x_1, data_y_1)
    flat_array, _, _ = borders_initializer(data_x_2, data_y_2)

    result_array = []
    for index in range(len(flat_array)):
        result_array.append(
            is_point_line(
                hallway_kern_down[0],
                hallway_kern_down[1],
                (data_x_2[index], flat_array[index]),
                lambda x, y: x >= y)
            and
            is_point_line(
                hallway_kern_up[0],
                hallway_kern_up[1],
                (data_x_2[index], flat_array[index]),
                lambda x, y: x <= y))

    result = np.array(result_array)
    if all(result):
        return True, None
    else:
        return False, result == False
class Connector_kern_cubes:

    def __init__(
            self,
            qa_qc_kern = None,
            qa_qc_cubes = None):
        assert not qa_qc_cubes or not qa_qc_kern, "[Connector_kern_cubes] Connector defined incorrectly"
        self.QA_QC_kern = qa_qc_kern
        self.QA_QC_cubes = qa_qc_cubes

    def kern_test_dependence_kpr_kp(self, poro_group_data: np.array, permX_group_data: np.array, cluster_key) -> (bool, np.array or None):
        kp = self.QA_QC_kern.porosity_open
        kpr = self.QA_QC_kern.kpr
        lithotype = self.QA_QC_kern.lithotype
        kp_clusters, kpr_cluster = CubesTools.get_cluster_dates(kp,kpr,lithotype)

        return check_data_point(kp_clusters[cluster_key], kpr_cluster[cluster_key], poro_group_data, permX_group_data)
