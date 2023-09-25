# Здесь реализуем классы для чтения всех требуемых форматов данных
import os
import sys

import numpy as np
import xtgeo
import fileinput


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

    def get_prop_value(self, prop: xtgeo.GridProperty, flag_d3: bool = True) -> np.array:
        if flag_d3:
            data = prop.get_npvalues3d()
        else:
            data = prop.get_npvalues1d()

        data[np.isnan(data)] = 0
        return data
    def get_grid(self) -> xtgeo.Grid:
        return self.grid
