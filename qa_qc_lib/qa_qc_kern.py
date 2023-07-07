import os
import shutil

import numpy as np


class QA_QC_kern():
    def __init__(self, filename="test_report.txt") -> None:
        """_summary_

        Args:
            data (str): _description_
        """
        self.filename = filename
        self.file = open(self.filename, "w")
        self.data = [
            0.052,
            0.076,
            0.018,
            0.119,
            0.138,
            0.156,
            0.16,
            0.165,
            0.138,
            0.159,
            0.133,
            0.159,
            0.147,
            0.146,
            0.084,
            0.097,
            0.125,
            0.135,
            90
        ]
        self.check_input()
        pass

    def check_input(self):
        if not isinstance(self, list):
            return False
        if len(self) == 0:
            return False
        for element in self:
            if not isinstance(element, (int, float)):
                return False

    def __del__(self):
        self.file.close()

    def test_water_saturation(self) -> bool:

        if all(0 <= x <= 1 for x in self.data):
            result = True
        else:
            result = False

        self.file.write("Test 'water_saturation': {}\n".format(result))
        return result

    def test_porosity(self) -> bool:
        if all(0 <= x <= 47.6 for x in self.data):
            result = True
        else:
            result = False
        self.file.write("Test 'porosity': {}\n".format(result))
        return result

    def test_permeability(self) -> bool:
        if all(x > 0 for x in self.data):
            result = True
        else:
            result = False
        self.file.write("Test 'permeability': {}\n".format(result))
        return result

    def test_monotony(self) -> bool:
        for i in range(1, len(self.data)):
            if self.data[i] <= self.data[i - 1]:
                result = False
        result = True
        self.file.write("Test 'monotony': {}\n".format(result))
        return result

    def get_list_of_tests(self) -> list:
        """_summary_

        Returns:
            list: _description_
        """

        test_methods = [method for method in dir(self) if
                        callable(getattr(self, method)) and method.startswith("test")]
        return test_methods

    def start_tests(self, list_of_tests: list) -> None:
        """_summary_

        Args:
            list_of_tests (list): _description_
        """
        for method_name in list_of_tests:
            method = getattr(self, method_name)
            method()
        pass

    def generate_test_report(self) -> str:
        """_summary_

        Returns:
            str: _description_
        """
        self.file.close()
        new_filepath = "D:\\" + self.filename  # Путь для сохранения файла на диске D:
        shutil.copy(self.filename, new_filepath)  # Копирование файла
        os.remove(self.filename)  # Удаление исходного файла
        return new_filepath
