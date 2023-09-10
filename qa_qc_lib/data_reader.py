# Здесь реализуем классы для чтения всех требуемых форматов данных

import sys

sys.path.append("../utils/GRDCL_parser")

from GRDECL_Parser import GRDECL_Parser
from GRDECL2VTK import GeologyModel
import fileinput
from os import listdir
from os.path import isfile, join

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


class QA_QC_grdecl_parser(object):
    def __init__(self, grid_file_path: str, file_path: str):
        model = GeologyModel(filename=grid_file_path)

        self.name_petrel = self.__get_name_in_petrel(file_path)
        print(model.GRDECL_Data.GRID_type)
        data = GRDECL_Parser(file_path,
                             model.GRDECL_Data.NX,
                             model.GRDECL_Data.NY,
                             model.GRDECL_Data.NZ,
                             model.GRDECL_Data.GRID_type,
                             False)

        print(data.SpatialDatas)
        model.GRDECL_Data.SpatialDatas = data.SpatialDatas

        self.model = model

    def Get_Model(self) -> [GeologyModel, str]:
        return self.model, self.name_petrel

    def __get_name_in_petrel(self, file_path: str) -> str:
        for line in open(file_path,'r').readlines():
            if "Property name in Petrel" in line:
                fileinput.close()
                return line.split(":")[1].strip()




test_class, name_petrel = QA_QC_grdecl_parser('../data/grdecl_data/GRID.GRDECL',
                                              '..\data/grdecl_data/input/So.GRDECL.grdecl').Get_Model()
print(test_class.GRDECL_Data.SpatialDatas[SupportTypePetrelDict[name_petrel]])
