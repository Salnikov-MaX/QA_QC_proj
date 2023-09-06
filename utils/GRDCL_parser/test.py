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

grid_file = 'input/grid_1.GRDECL'
replaceAll(grid_file,'-- Generated : Petrel'," ")
model_grid_data_2 = GeologyModel(filename=grid_file)

swl_file = 'input/directly/Poro1_3.GRDECL'
replaceAll(swl_file,'-- Generated : Petrel'," ")

swl_data = GRDECL_Parser_my(filename=swl_file, grid_data=False, GRID_type=model_grid_data_2.GRDECL_Data.GRID_type, grid_dim=[
                                    model_grid_data_2.GRDECL_Data.NX, model_grid_data_2.GRDECL_Data.NY, model_grid_data_2.GRDECL_Data.NZ])

model_grid_data_2.GRDECL_Data.SpatialDatas = swl_data.PORO

result = model_grid_data_2.GRDECL_Data.SpatialDatas

np.set_printoptions(threshold=sys.maxsize)

print(len(result))
