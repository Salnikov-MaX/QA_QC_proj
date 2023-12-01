# Здесь реализуем классы для чтения всех требуемых форматов данных
import os

import numpy as np
import pandas as pd
import xtgeo
from qa_qc_lib.tools.cubes_tools import CubesTools


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

        return data
    def get_grid(self) -> xtgeo.Grid:
        return self.grid

class QA_QC_asciigrid_parser(object):
    def __init__(self):
        pass

    def parse_to_dataframe(self, filepath: str):
        head = ""
        xarray = []
        yarray = []
        zarray = []
        with open(filepath, 'r') as f:
            content = f.readlines()
            for line in content:
                if '#' in line:
                    head += line
                    continue
                l = line.split(' ')
                xarray.append(float(l[0]))
                yarray.append(float(l[1]))
                zarray.append(float(l[2]))

        df = pd.DataFrame({'x': xarray, 'y': yarray, 'z': zarray})
        return df, head

    def get_pologin(self, df: pd.DataFrame):
        return xtgeo.Polygons(values=np.array(df))

def test():
    test = QA_QC_grdecl_parser("../data/grdecl_data","GRID")
    poro_file = "../data/grdecl_data/input/Poro.GRDECL.grdecl"
    flag, key = CubesTools().find_key(poro_file)
    test.add_prop(poro_file, key)
    prop_value = test.get_prop_value(test.get_grid().get_prop_by_name(key),False)
    print(prop_value)

def ascii_test():
    df, head = QA_QC_asciigrid_parser().parse_to_dataframe("../../data/grdecl_data/asciigrid/ГНК_ASCIIGRID_ПЕТРОФИЗИКА_.txt")
    pol = QA_QC_asciigrid_parser().get_pologin(df)
    print(pol.dataframe)