# Здесь реализуем классы для чтения всех требуемых форматов данных
import os
import sys

import numpy as np
import xtgeo
import fileinput
from qa_qc_lib.qa_qc_tools.cubes_tools import CubesTools


class QA_QC_grdecl_parser(object):
    def __init__(self, directory_path: str, grid_name: str):
        file_name = ["", "_ACTNUM", "_COORD", "_ZCORN"]
        with open(directory_path + "/" + "temporary.GRDECL", 'w') as outfile:
            for fname in file_name:
                with open(directory_path + "/" + grid_name + fname + ".GRDECL") as infile:
                    outfile.write(infile.read())
                    infile.close()
            outfile.close()

        self.grid = xtgeo.grid_from_file(directory_path + "/" + "temporary.GRDECL", fformat="grdecl")
        os.remove(directory_path + "/" + "temporary.GRDECL")

    def add_prop(self, file_path: str, prop_name: str):
        self.grid.append_prop(xtgeo.gridproperty_from_file(
            file_path,
            name=prop_name,
            grid=self.grid,
        ))

    def get_prop_value(self, prop: xtgeo.GridProperty) -> np.array:
        return prop.get_npvalues3d()
    def get_grid(self) -> xtgeo.Grid:
        return self.grid

    def generate_wrong_actnum(self,wrong_actnum: xtgeo.GridProperty, head: str = "",save_path: str = '.', func_name:str = "QA/QC"):
        result_data = f"{head}\n-- Generated QA/QC\n"

        with open(f"{save_path}/{func_name}_WRONG_ACTNUM.GRDECL", 'w') as f:
            f.write(result_data)
            f.close()

        wrong_actnum.to_file(
            pfile=f"{save_path}/{func_name}_WRONG_ACTNUM.GRDECL",
            fformat="grdecl",
            name="ACTNUM",
            append=True)

        print(f"Файл WRONG_ACTNUM сохранён по пути: {save_path}")

def test():
    test = QA_QC_grdecl_parser("../data/grdecl_data","GRID")
    poro_file = "../data/grdecl_data/input/Poro.GRDECL.grdecl"
    flag, key = CubesTools().find_key(poro_file)
    test.add_prop(poro_file, key)
    prop_value = test.get_prop_value(test.get_grid().get_prop_by_name(key))
    np.set_printoptions(threshold=np.inf)
    print(prop_value)