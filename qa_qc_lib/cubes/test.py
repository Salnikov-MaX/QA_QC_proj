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

#main()

def RemoveCommentLines(data, commenter='--'):
    # Remove comment and empty lines
    data_lines = data.strip().split('\n')
    newdata = []
    for line in data_lines:
        if not line.strip():
            continue
        elif line.find(commenter) != -1:
            newline = line[0:line.find(commenter)].strip()
            if len(newline) == 0:
                continue
            newdata.append(newline)
        else:
            newdata.append(line)
    return '\n'.join(newdata)

with open("../../data/grdecl_data/input/Sgu.GRDECL.grdecl","r") as f:
    contents = f.read()
    contents = RemoveCommentLines(contents, commenter='--')
    contents_in_block = contents.strip().split('/')
    contents_in_block = [x for x in contents_in_block if x]
    NumKeywords = len(contents_in_block)
    if NumKeywords > 2 or NumKeywords == 0:
        print(False)
    else:
        print(contents_in_block[NumKeywords-1].split()[0])