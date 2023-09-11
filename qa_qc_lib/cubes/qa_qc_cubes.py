from qa_qc_lib.data_reader import QA_QC_grdecl_parser
from qa_qc_lib.qa_qc_main import QA_QC_main, Type_Status
import numpy as np
import datetime

SupportTypePetrelDict = {
    "Permeability_NP4": "PERMX",
    "Porosity_NP4": "PORO",
    "Sg": "SGAS",
    "Sgcr": "SGCR",
    "SGL": "IRR.GASSATURATION",
    "Sgu": "SGAS",
    "So": "SOIL",
    "SOWCR": "CRITICALOILSATURATION",
    "Sw": "SWAT",
    "Swl": "IRR.WATERSATURATION",
    "SWU": "SWAT",
    "OWC_NP4": "ABOVECONTACT",
    "GOC_NP4": "ABOVECONTACT",
}


class QA_QC_cubes(QA_QC_main):

    def __init__(self, grid_path: str, logging_setting_path="../../report", logging_setting_file_name=None):
        QA_QC_main.__init__(self)
        self.grid_file = grid_path

    def __pars_wrong_values(self, wrong_list: list[
        list[int or float],
        list[int]
    ]) -> str:
        result = ""
        for i in range(len(wrong_list[0])):
            result += f"value:{wrong_list[0][i]} " + f"index: {wrong_list[1][i]}\n"
        return result

    def __test_range_data(self, array: np.array, lambda_list: list[any]) -> tuple[
        bool,
        list[
            list[int or float],
            list[int]
        ] or None]:

        mask_array = (sum(func(array) for func in lambda_list)).astype(dtype=bool)

        if not any(mask_array):
            return True, None
        else:
            return False, [array[mask_array].tolist(), np.where(mask_array == True)[0].tolist()]

    """
    Тесты первого порядка
    """

    def test_open_porosity(self, file_path: str):
        data, key_petrel, err = QA_QC_grdecl_parser(self.grid_file, file_path=file_path).Get_Model()
        if err is not None:
            self.update_report(self.generate_report_text(err.message, Type_Status.NotRunning))
            return

        flag, wrong_data = self.__test_range_data(
            data.GRDECL_Data.SpatialDatas[SupportTypePetrelDict[key_petrel]],
            [lambda x: x < 0, lambda x: x > 0.476])

        if flag:
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
        else:
            self.update_report(self.generate_report_text(f"Данные \n {self.__pars_wrong_values(wrong_data)}  лежат не в интервале от 0 до 47,6", Type_Status.NotPassed))

        self.generate_test_report(file_name=self.__class__.__name__, file_path="../../report", data_name=file_path)

    def test_permeability(self, file_path: str):
        data, key_petrel, err = QA_QC_grdecl_parser(self.grid_file, file_path=file_path).Get_Model()
        if err is not None:
            self.update_report(self.generate_report_text(err.message, Type_Status.NotRunning))
            return

        flag, wrong_data = self.__test_range_data(
            data.GRDECL_Data.SpatialDatas[SupportTypePetrelDict[key_petrel]],
            [lambda x: x < 0])

        if flag:
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
        else:
            self.update_report(self.generate_report_text(
                f"Данные \n {self.__pars_wrong_values(wrong_data)}  < 0",
                Type_Status.NotPassed))

    def test_range_data(self, file_path: str):
        data, key_petrel, err = QA_QC_grdecl_parser(self.grid_file, file_path=file_path).Get_Model()
        if err is not None:
            self.update_report(self.generate_report_text(err.message, Type_Status.NotRunning))
            return

        flag, wrong_data = self.__test_range_data(
            data.GRDECL_Data.SpatialDatas[SupportTypePetrelDict[key_petrel]],
            [lambda x: x < 0, lambda x: x > 1])

        if flag:
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
        else:
            self.update_report(self.generate_report_text(
                f"Данные \n {self.__pars_wrong_values(wrong_data)}  лежат не в интервале от 0 до 1 ",
                Type_Status.NotPassed))

        self.generate_test_report(file_name=self.__class__.__name__, file_path="../../report", data_name=file_path)


    def test_integer_data(self, file_path: str):
        data, key_petrel, err = QA_QC_grdecl_parser(self.grid_file, file_path=file_path).Get_Model()
        if err is not None:
            self.update_report(self.generate_report_text(err.message, Type_Status.NotRunning))
            return

        flag, wrong_data = self.__test_range_data(
            data.GRDECL_Data.SpatialDatas[SupportTypePetrelDict[key_petrel]],
            [lambda x: x == 0, lambda x: x == 1])

        if flag:
            self.update_report(self.generate_report_text("", Type_Status.Passed.value))
        else:
            self.update_report(self.generate_report_text(
                f"Данные \n {self.__pars_wrong_values(wrong_data)}  не целочисленные",
                Type_Status.NotPassed))

        self.generate_test_report(file_name=self.__class__.__name__, file_path="../../report", data_name=file_path)
QA_QC_cubes("../../data/grdecl_data/GRID.GRDECL").test_open_porosity("../../data/grdecl_data/input/Poro.GRDECL.grdecl")
