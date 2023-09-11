# Здесь реализуем классы для чтения всех требуемых форматов данных

import sys

from qa_qc_lib.cubes.GRDCL_parser.GRDECL_Parser import GRDECL_Parser
from qa_qc_lib.cubes.GRDCL_parser.GRDECL2VTK import GeologyModel
import fileinput
from qa_qc_lib.error import Error, Type_Error


class QA_QC_grdecl_parser(object):
    def __init__(self, grid_file_path: str, file_path: str):
        self.error = None
        self.model = None
        self.name_petrel = None

        model = GeologyModel(filename=grid_file_path)
        if model.GRDECL_Data.error is not None:
            self.error = model.GRDECL_Data.error
            return

        self.name_petrel, err = self.__get_name_in_petrel(file_path)
        if err is not None:
            self.error = err
            return

        data = GRDECL_Parser(file_path,
                             model.GRDECL_Data.NX,
                             model.GRDECL_Data.NY,
                             model.GRDECL_Data.NZ,
                             model.GRDECL_Data.GRID_type,
                             False)

        if data.error is not None:
            self.error = data.error
            return

        model.GRDECL_Data.SpatialDatas = data.SpatialDatas

        self.model = model

    def Get_Model(self) -> [GeologyModel, str, Error or None]:
        return self.model, self.name_petrel, self.error

    def __get_name_in_petrel(self, file_path: str) -> tuple[str, Error or None]:
        for line in open(file_path, 'r').readlines():
            if "Property name in Petrel" in line:
                fileinput.close()
                return line.split(":")[1].strip(), None

        return "", Error(type_error=Type_Error.pars_error, message="Property name in Petrel отсутсвует в файле")
