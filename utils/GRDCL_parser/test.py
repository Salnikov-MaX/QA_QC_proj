import numpy as np

from GRDECL_Parser_my import GRDECL_Parser_my
from GRDECL2VTK import GeologyModel
import fileinput
import sys
def replaceAll(file, searchExp, replaceExp):
    for line in fileinput.input(file, inplace=1):
        if searchExp in line:
            line = line.replace(searchExp,replaceExp)
        sys.stdout.write(line)

grid_file = '../../data/grdecl_data/GRID.GRDECL'
replaceAll(grid_file,'-- Generated : Petrel'," ")
model_grid_data_2 = GeologyModel(filename=grid_file)

swl_file = '../../data/grdecl_data/input/Swl.GRDECL.grdecl'
replaceAll(swl_file,'-- Generated : Petrel'," ")
replaceAll(swl_file,'IRR.WATERSATURATION',"SW_NPSL")

swl_data = GRDECL_Parser_my(filename=swl_file, grid_data=False, GRID_type=model_grid_data_2.GRDECL_Data.GRID_type, grid_dim=[
                                    model_grid_data_2.GRDECL_Data.NX, model_grid_data_2.GRDECL_Data.NY, model_grid_data_2.GRDECL_Data.NZ])

print(swl_data.TypePetrel)
model_grid_data_2.GRDECL_Data.SpatialDatas = swl_data.SW_NPSL

result = model_grid_data_2.GRDECL_Data.SpatialDatas

print(result)
