import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Alignment
from openpyxl.utils import get_column_letter
from qa_qc_kern import QA_QC_kern


class DataPreprocessing:
    def __init__(self, files=None, glossary_of_names=None):
        if glossary_of_names is None:
            glossary_of_names = {}
        if files is None:
            files = []
        self.wb = Workbook()
        self.ws = self.wb.active
        self.user_glossary_of_names = glossary_of_names
        self.input_files = files
        self.failed_tests = {}
        self.glossary_of_names = {
            "Лабораторный номер": "",
            "Кровля интервала отбора": "",
            "Подошва интервала отбора": "",
            "Место отбора (ниже кровли), м": "",
            "Глубина отбора, м": "",
            "Вынос керна, м": "",
            "Вынос керна, %": "",
            "Ск %": "",
            "Открытая пористость по жидкости": "",
            "Открытая пористость по газу": "",
            "Открытая пористость в пластовых условиях": "",
            "Открытая пористость по керосину": "",
            "Эффективная пористость": "",
            "Динамическая пористость": "",
            "Кпр_газ(гелий)": "",
            "Параметр пористости": "",
            "So": "",
            "Кво": "",
            "Плотность абсолютно сухого образца": "",
            "Рн": "",
            "Sw": "",
            "Газопроницаемость по Кликенбергу": "",
            "Газопроницаемость, mkm2 (parallel)": "",
            "Газопроницаемость Кликенбергу": "",
            "Объемная плотность": "",
            "Минералогическая плотность": "",
            "Эффективная проницаемость": "",
            "Ск": "",
            "Параметр насыщения": "",
            "Газопроницаемость эфф": "",
            "Газопроницаемость по воде": "",
            "Плотность максимально увлажненного образца": "",
            "Упругие свойства": "",
            "Примечание": ""
        }
        self.data_dict = {
            "Лабораторный номер": None,
            "Кровля интервала отбора": None,
            "Подошва интервала отбора": None,
            "Место отбора (ниже кровли), м": None,
            "Глубина отбора, м": None,
            "Вынос керна, м": None,
            "Вынос керна, %": None,
            "Ск %": None,
            "Открытая пористость по жидкости": None,
            "Открытая пористость по газу": None,
            "Открытая пористость в пластовых условиях": None,
            "Открытая пористость по керосину": None,
            "Эффективная пористость": None,
            "Динамическая пористость": None,
            "Кпр_газ(гелий)": None,
            "Параметр пористости": None,
            "Эффективная проницаемость": None,
            "So": None,
            "Газопроницаемость по Кликенбергу": None,
            "Кво": None,
            "Параметр насыщения": None,
            "Плотность абсолютно сухого образца": None,
            "Рн": None,
            "Sw": None,
            "Газопроницаемость, mkm2 (parallel)": None,
            "Газопроницаемость Кликенбергу": None,
            "Объемная плотность": None,
            "Минералогическая плотность": None,
            "Ск": None,
            "Газопроницаемость эфф": None,
            "Газопроницаемость по воде": None,
            "Плотность максимально увлажненного образца": None,
            "Упругие свойства": None,
            "Примечание": None
        }
        self.__create_new_glossary()
        self.headers = [
            "Лабораторный номер", "Кровля интервала отбора", "Подошва интервала отбора",
            "Эффективная проницаемость", "Параметр насыщения",
            "Место отбора (ниже кровли), м", "Глубина отбора, м",
            "Вынос керна, м", "Вынос керна, %", "Ск %", "Открытая пористость по жидкости",
            "Открытая пористость по газу",
            "Открытая пористость в пластовых условиях", "Открытая пористость по керосину", "Эффективная пористость",
            "Динамическая пористость", "Кпр_газ(гелий)", "Параметр пористости", "So",
            "Кво", "Плотность абсолютно сухого образца", "Рн",
            "Sw", "Газопроницаемость, mkm2 (parallel)", "Газопроницаемость Кликенбергу",
            "Объемная плотность", "Минералогическая плотность", "Ск",
            "Газопроницаемость эфф", "Газопроницаемость по воде",
            "Плотность максимально увлажненного образца"
            "Упругие свойства",
            "Примечание"
        ]

    def __create_new_glossary(self):
        for key in self.user_glossary_of_names:
            if key in self.glossary_of_names:
                self.glossary_of_names[key] = self.user_glossary_of_names[key]

    def file_creation(self):
        # Создаем Excel-таблицу с заголовками
        for idx, header in enumerate(self.headers, start=1):
            self.ws.cell(row=1, column=idx, value=header)
            if len(header) > 7:
                letter = get_column_letter(idx)
                self.ws.column_dimensions[letter].width = 20

        df = []
        for file in self.input_files:
            if file.endswith(".xlsx") or file.endswith(".xls"):
                df = pd.read_excel(file)
            elif file.endswith(".txt"):
                df = pd.read_csv(file, sep="\t")
            if len(df) != 0:
                for col in df.columns:
                    if col in self.glossary_of_names.values():
                        key = list(self.glossary_of_names.keys())[list(self.glossary_of_names.values()).index(col)]
                        col_data = df[col].tolist()
                        processed_col_data = []
                        for value in col_data:
                            try:
                                processed_value = float(value.replace(',', '.'))
                            except:
                                processed_value = value
                            processed_col_data.append(processed_value)
                        self.data_dict[key] = processed_col_data
        print(self.data_dict)
        for idx, (key, values) in enumerate(self.data_dict.items(), start=2):
            if values is not None:
                for row_idx, value in enumerate(values, start=2):
                    col_idx = self.headers.index(key) + 1
                    self.ws.cell(row=row_idx, column=col_idx, value=value)
        return self.data_dict

    def start_tests(self, tests_name=None):
        if tests_name is None:
            tests_name = []
        test_system = QA_QC_kern(pas=np.array(self.data_dict["Плотность абсолютно сухого образца"]),
                                 note=np.array(self.data_dict["Примечание"]),
                                 kno=np.array(self.data_dict["So"]),
                                 kp_plast=np.array(self.data_dict["Открытая пористость в пластовых условиях"]),
                                 density=np.array(self.data_dict["Плотность абсолютно сухого образца"]),
                                 water_permeability=np.array(self.data_dict["Газопроницаемость по воде"]),
                                 kp_pov=np.array(self.data_dict["Открытая пористость по жидкости"]),
                                 # perpendicular=None, - предобработать
                                 # perpendicular_density - предобработать
                                 kp=np.array(self.data_dict["Открытая пористость по жидкости"]),
                                 roof=(self.data_dict["Кровля интервала отбора"]),
                                 core_removal_in_meters=(self.data_dict["Вынос керна, м"]),
                                 # parallel_carbonate=None, - предобработать
                                 # perpendicular_carbonate=None, - предобработать
                                 # perpendicular_porosity=None, -предобработать
                                 # intervals=None, - предобработать
                                 sole=(self.data_dict["Подошва интервала отбора"]),
                                 percent_core_removal=(self.data_dict["Вынос керна, %"]),
                                 outreach_in_meters=(self.data_dict["Вынос керна, м"]),
                                 sw_residual=(self.data_dict["Кво"]),
                                 core_sampling=(self.data_dict["Глубина отбора, м"]),
                                 kpr=(self.data_dict["Кпр_газ(гелий)"]),
                                 # parallel_density=None, -предобработка
                                 # parallel_porosity=None, - предобработка
                                 # parallel=(self.data_dict["Глубина отбора, м"]), -предобработка
                                 rp=(self.data_dict["Параметр пористости"]),
                                 pmu=(self.data_dict["Плотность максимально увлажненного образца"]),
                                 rn=(self.data_dict["Параметр насыщения"]),
                                 obplnas=(self.data_dict["Плотность абсолютно сухого образца"]),
                                 poroTBU=(self.data_dict["Открытая пористость в пластовых условиях"]),
                                 poroHe=(self.data_dict["Открытая пористость по газу"]),
                                 porosity_open=(self.data_dict["Открытая пористость по жидкости"]),
                                 porosity_kerosine=(self.data_dict["Открытая пористость по керосину"]),
                                 sw=(self.data_dict["Sw"]),
                                 parallel_permeability=(self.data_dict["Газопроницаемость, mkm2 (parallel)"]),
                                 klickenberg_permeability=(self.data_dict["Газопроницаемость по Кликенбергу"]),
                                 effective_permeability=(self.data_dict["Эффективная проницаемость"]),
                                 md=(self.data_dict["Место отбора (ниже кровли), м"]), )

        self.failed_tests = test_system.start_tests(tests_name)["wrong_parameters"]
        test_system.generate_test_report()
        self.error_flagging()
        return self.failed_tests

    def error_flagging(self):
        self.ws.cell(row=1, column=33, value="Примечание")
        self.ws.merge_cells(start_row=1, start_column=33, end_row=1,
                            end_column=38)
        self.ws.cell(row=1, column=33).alignment = Alignment(horizontal="center",
                                                             vertical="center")
        for idx, (test_name, failed_columns) in enumerate(self.failed_tests.items(), start=2):
            self.ws.cell(row=2, column=33 + idx + 1, value=test_name)
            letter = get_column_letter(33 + idx + 1)
            self.ws.column_dimensions[letter].width = 30
            print(failed_columns.items())
            for col, failed_indices in failed_columns.items():
                col_idx = self.headers.index(col) + 1  # Index of the column in the headers
                for row_idx in failed_indices[0]:
                    cell = self.ws.cell(row=row_idx + 2, column=col_idx)
                    cell.fill = PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")
                    self.ws.cell(row=row_idx + 3, column=33 + idx + 1, value=failed_indices[1])

    def save_file(self, output_excel_path="C:\\Users\\nikit\\Downloads", file_name="excel"):
        self.wb.save(f"{output_excel_path}\\{file_name}.xls")


input_files = ["C:\\Users\\nikit\\Downloads\\Poro.txt",
               "C:\\Users\\nikit\\Downloads\\Примечение.txt",
               "C:\\Users\\nikit\\Downloads\\59PObraz.xlsx"]
dic = {
    "Эффективная проницаемость": "Prohodka, m",
    "Лабораторный номер": "Number",
    # "Кровля интервала отбора": "Top",
    # "Подошва интервала отбора": "Bottom",
    # "Место отбора (ниже кровли), м": "Mesto otbora",
    # "Глубина отбора, м": "MD",
    # "Вынос керна, м": "PoroTBU",
    # "Вынос керна, %": "Vynos, m",
    # "Ск %": "Carbonate",
    # "Открытая пористость по жидкости": "Porosity (open)",
    # "Открытая пористость по газу": "From",
    # "Открытая пористость в пластовых условиях": "Porosity (total)",
    # "Открытая пористость по керосину": "Porosity (kerosine)",
    # "Эффективная пористость": "Porosity (effective)",
    # "Динамическая пористость": "Permeability, mkm2 (parallel)",
    # "Кпр_газ(гелий)": "Permeability (perpendicular)",
    # "Параметр пористости": "So",
    # "So": "FF",
    # "Кво": "Vs, km/s",
    # "Плотность абсолютно сухого образца": "Density, g/cc",
    # "Рн": "PoroTBU",
    # "Sw": "Sw",
    # "Газопроницаемость по Кликенбергу": "PoroTBU",
    # "Газопроницаемость, mkm2 (parallel)": "PoroTBU",
    # "Газопроницаемость Кликенбергу": "PoroTBU",
    # "Объемная плотность": "PoroTBU",
    # "Минералогическая плотность": "PoroTBU",
    # "Параметр насыщения": "PoroTBU",
    # "Газопроницаемость эфф": "PoroTBU",
    # "Газопроницаемость по воде": "PoroTBU",
    # "Плотность максимально увлажненного образца": "PoroTBU",
    # "Упругие свойства": "PoroTBU",
    # "Примечание": "Примечание"
    }

file_modal = DataPreprocessing(files=input_files, glossary_of_names=dic)
file_modal.file_creation()
# print(file_modal.start_tests(["test_table_notes"]))
file_modal.save_file()
