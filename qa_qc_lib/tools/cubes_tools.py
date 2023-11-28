#########################################################################################################
#######################| РЕАЛИЗАЦИЯ ВСПОМОГАТЕЛЬНЫХ ФУНКЦИЙ QA_QC_cubes |##############################
#########################################################################################################

import numpy as np
import inspect

class CubesTools:
    def __init__(self):
        pass

    def __remove_comment_lines(self, data, commenter='--') -> str:
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

    def find_key(self, file_path: str) -> [bool, str]:
        with open(file_path, 'r') as f:
            contents = f.read()
            contents = self.__remove_comment_lines(contents, commenter='--')
            contents_in_block = contents.strip().split('/')
            contents_in_block = [x for x in contents_in_block if x]
            NumKeywords = len(contents_in_block)
            if NumKeywords > 2 or NumKeywords == 0:
                return False, ""
            else:
                return True, contents_in_block[NumKeywords - 1].split()[0]

    def find_head(self, file_path:str) -> str:
        head = ""
        with open(file_path, 'r') as f:
            contants = f.readlines()
            readflag = False
            for line in contants:
                if '[' in line:
                    readflag = True

                if readflag:
                    head += f"{line.strip()}\n"

                if ']' in line:
                    break

        return head

    def generate_wrong_actnum(self,wrong_list: np.array, head: str = "",save_path: str = '.', func_name:str = "QA-QC"):
        wrong_data = wrong_list.astype(dtype=int)
        result_data = f"{head}\n-- Generated QA/QC\nACTNUM\n"
        counter = 0
        for index in range(len(wrong_data) - 2):
            if wrong_data[index] == wrong_data[index + 1]:
                counter += 1
            else:
                if counter != 0:
                    result_data += f"{counter + 1}*{wrong_data[index]} "
                    counter = 0
                else:
                    result_data += f"{wrong_data[index]} "

        if wrong_data[len(wrong_data) - 1] == wrong_data[len(wrong_data) - 2]:
            result_data += f"{counter + 2}*{wrong_data[len(wrong_data) - 1]} \\"
        else:
            s = ""
            if counter == 0:
                s += f"{wrong_data[len(wrong_data) - 2]}"
            else:
                s += f"{counter + 1}*{wrong_data[len(wrong_data) - 2]}"
            result_data += f"{s} {wrong_data[len(wrong_data) - 1]} \\"

        with open(f"{save_path}/{func_name}_WRONG_ACTNUM.GRDECL", 'w') as f:
            f.write(result_data)
            f.close()

        print(f"Файл WRONG_ACTNUM сохранён по пути: {save_path}")

    def generate_wrong_map(self,wrong_list: np.array, head: str = "",save_path: str = '.', func_name:str = "QA-QC"):
        wrong_data = wrong_list.astype(dtype=int)
        result_data = f"{head}\n# Generated QA/QC\n"
        for index in wrong_data:
            result_data += "0 0 " + str(index) + " 0 0\n"

        with open(f"{save_path}/{func_name}_WRONG_MAP.txt", 'w') as f:
            f.write(result_data)
            f.close()

        print(f"Файл WRONG_MAP сохранён по пути: {save_path}")
    def get_cluster_dates(self, data1, data2, lit_data):
        litatype_unique_data = np.unique(lit_data)
        return {value: data1[np.where(lit_data == value)] for value in litatype_unique_data}, {
            value: data2[np.where(lit_data == value)] for value in litatype_unique_data}

    def conver_n3d_to_n1d(self, cube):
        return np.ravel(cube, order='C').reshape((1, -1))[0]

    def conver_n1d_to_n3d(self, grid, vector):
        i_max = grid.ncol
        j_max = grid.nrow
        k_max = grid.nlay
        return np.reshape(vector, (i_max, j_max, k_max), order='C')

