from qa_qc_lib.data_reader import QA_QC_grdecl_parser
import numpy as np
import datetime
import logging

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

"""
class Report_Creator():

    def __init__(self, file_report_path: str, req_reports: list[bool,bool,bool]):

    def
"""

class CustomAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        context = kwargs.pop('file', self.extra['file'])
        return '%s | [%s]' % (msg, context), kwargs

class QA_QC_cubes(object):

    def __init__(self, grid_path: str, logging_setting_path="../../report", logging_setting_file_name= None):
        self.grid_file = grid_path
        if logging_setting_file_name is None:
            self.logger = self.__setup_logging(logging_setting_path, self.__class__.__name__)
        else:
            self.logger = self.__setup_logging(logging_setting_path, logging_setting_file_name)

        self.logger.info("Start module QA_QC_cubes")

    def __setup_logging(self, path: str, file_name: str) -> logging:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
        logging.basicConfig(
            filename=f"{path}/{timestamp}-{file_name}.log",
            format="%(asctime)s - %(module)s - %(levelname)s - %(funcName)s: %(message)s",
            datefmt='%H:%M:%S',
        )

        return CustomAdapter(logging.getLogger(self.__class__.__name__), {'file': None})

    def __pars_wrong_values(self, wrong_list: list[
        list[int or float],
        list[int]
    ]) -> str:
        result = ""
        for i in range(len(wrong_list[0])):
            result += f"value:{wrong_list[0][i]} " + f"index: {wrong_list[1][i]}\n"
        return result

    def __test_porosity(self, array: np.array) -> tuple[
        bool,
        list[
            list[int or float],
            list[int]
        ] or None]:

        mask_array = (array < 0) + (array > 0.476)

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
            self.logger.error(err.message)
            return

        flag, wrong_data = self.__test_porosity(data.GRDECL_Data.SpatialDatas[SupportTypePetrelDict[key_petrel]])
        if flag:
            self.logger.info("OK")
        else:
            self.logger.error(f"Данные \n {self.__pars_wrong_values(wrong_data)}  лежат не в интервале от 0 до 47,6")


test = QA_QC_cubes("../../data/grdecl_data/GRID.GRDECL").test_open_porosity("../../data/grdecl_data/input/Poro.GRDECL.grdecl")