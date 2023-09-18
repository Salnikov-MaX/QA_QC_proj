import os

import xtgeo
from qa_qc_lib.data_reader import QA_QC_grdecl_parser

#model = QA_QC_grdecl_parser(directory_path="../../data/grdecl_data", grid_name="GRID").Get_Model()
directory_path = "../../data/grdecl_data"
grid_name = "GRID"

file_name = ["", "_ACTNUM", "_COORD", "_ZCORN"]
with open(directory_path + "/" + "temporary.GRDECL", 'w') as outfile:
    for fname in file_name:
        with open(directory_path + "/" + grid_name + fname + ".GRDECL") as infile:
            outfile.write(infile.read())
            infile.close()
    outfile.close()

test = xtgeo.grid_from_file(directory_path + "/" + "temporary.GRDECL", fformat="grdecl")
os.remove(directory_path + "/" + "temporary.GRDECL")