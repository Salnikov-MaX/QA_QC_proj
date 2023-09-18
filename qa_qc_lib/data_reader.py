# Здесь реализуем классы для чтения всех требуемых форматов данных
import os
import sys

from qa_qc_lib.cubes.GRDCL_parser.GRDECL_Parser import GRDECL_Parser
from qa_qc_lib.cubes.GRDCL_parser.GRDECL2VTK import GeologyModel
import fileinput


class QA_QC_grdecl_parser(object):
    def __init__(self, directory_path: str, grid_name: str):
        model = GeologyModel(filename=directory_path + "/" + grid_name + ".GRDECL")
        file_name = ["", "_ACTNUM", "_COORD", "_ZCORN"]
        with open(directory_path + "/" + "temporary.GRDECL", 'w') as outfile:
            for fname in file_name:
                with open(directory_path + "/" + grid_name + fname + ".GRDECL") as infile:
                    outfile.write(infile.read())
                    infile.close()
            outfile.close()

        model.GRDECL_Data = GRDECL_Parser(directory_path + "/" + "temporary.GRDECL",
                                          model.GRDECL_Data.NX,
                                          model.GRDECL_Data.NY,
                                          model.GRDECL_Data.NZ,
                                          model.GRDECL_Data.GRID_type,
                                          False)
        os.remove(directory_path + "/" + "temporary.GRDECL")

        self.model = model

    def Get_Model(self) -> [GeologyModel]:
        return self.model