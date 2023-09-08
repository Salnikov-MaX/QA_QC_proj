# Здесь реализуем классы для чтения всех требуемых форматов данных

import sys

sys.path.append("../utils/GRDCL_parser")

from GRDECL_Parser_my import GRDECL_Parser_my
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

SupportReplaceDict = {
    "SGL": "IRR.GASSATURATION",
    "Sgu": "SGAS_U",
    "Swl": "IRR.WATERSATURATION",
    "SWU": "SWAT_U",
    "OWC_NP4": "ABOVECONTACT",
    "GOC_NP4": "ABOVECONTACT",
}
class QA_QC_grdecl_parser(object):
    def __init__(self, grid_file_path: str, directory_path: str) -> GeologyModel:
        grid_file = grid_file_path
        self.__replace_all(grid_file,'-- Generated : Petrel'," ")
        model = GeologyModel(filename=grid_file)
        for file in self.__get_list_files(directory_path):
            file_path = directory_path +"/"+ file
            name_petrel = self.__get_name_in_petrel(file_path)
            print(name_petrel)
            data = GRDECL_Parser_my(filename=file_path,
                                    grid_data=False,
                                    GRID_type=model.GRDECL_Data.GRID_type,
                                    grid_dim=[
                                        model.GRDECL_Data.NX,
                                        model.GRDECL_Data.NY,
                                        model.GRDECL_Data.NZ
                                    ])

            print(data.KeyData)
            model.GRDECL_Data.SpatialDatas[name_petrel] = data.KeyData[SupportTypePetrelDict[name_petrel]]

        return model

    def __get_list_files(self,directory_path: str) -> list:
        onlyfiles = [f for f in listdir(directory_path) if isfile(join(directory_path, f))]
        return onlyfiles[:]
    def __replace_all(self,file: str, searchExp: str, replaceExp: str) -> None:
        for line in fileinput.input(file, inplace=1):
            if searchExp in line:
                line = line.replace(searchExp, replaceExp)
            sys.stdout.write(line)
    def __get_name_in_petrel(self,file_path: str) ->str:
        for line in fileinput.input(file_path, inplace=1):
            if "Property name in Petrel" in line:
                return line.split(":")[1].strip()

test_class = QA_QC_grdecl_parser('../data/grdecl_data/GRID.GRDECL','..\data\grdecl_data\input')
print(test_class.GRDECL_Data.SpatialDatas["SGCR"])