import os

import numpy as np
import xtgeo


def main():
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

    p = test.get_bulk_volume(asmasked=False)

    print(test.actnum_array)

main()
