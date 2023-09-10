import os
import pandas as pd
import numpy as np
from qa_qc_lib.qa_qc_kern import QA_QC_kern
from qa_qc_lib.qa_qc_tools.kern_tools import find_test_methods_with_params


class DataPreprocessing:
    def __init__(self):
        self.failed_tests = {}
        self.headers = [
            "Лабораторный номер", "Кровля интервала отбора", "Подошва интервала отбора",
            "Эффективная проницаемость", "Параметр насыщения",
            "Место отбора (ниже кровли), м", "Глубина отбора, м",
            "Вынос керна, м", "Вынос керна, %", "Ск %", "Открытая пористость по жидкости",
            "Открытая пористость по газу",
            "Открытая пористость в пластовых условиях", "Открытая пористость по керосину", "Кпр_газ(гелий)",
            "Параметр пористости", "So",
            "Кво", "Плотность абсолютно сухого образца", "Рн",
            "Sw", "Газопроницаемость, mkm2 (parallel)", "Газопроницаемость по Кликенбергу",
            "Объемная плотность", "Минералогическая плотность", "Ск", "Газопроницаемость по воде",
            "Плотность максимально увлажненного образца","Эффективная пористость",
            "Упругие свойства","Критическая нефтенасыщенность","Sgl","Sogcr "
            "Примечание", "Направление"
        ]
        self.parallel_density = []
        self.parallel_porosity = []
        self.parallel_number = []
        self.perpendicular_density = []
        self.perpendicular_porosity = []
        self.perpendicular_number = []
        self.parallel_carbonate = []
        self.interval = []
        self.perpendicular_carbonate = []
        self.df_result = []
        self.newdic = {}

    def get_possible_tests(self, columns_with_data):
        '''
        Опрделяет список тестов, которые возможно провести с текущеми данными

        :param columns_with_data: array[string]  массив с названиями не пустых колонок
        :return: array[string] - массив возможных тестов
        '''
        test = find_test_methods_with_params(columns_with_data, QA_QC_kern())
        return test

    def process_data(self, columns_mapping):
        '''
        Проходится по файлам и из каждого файла берет нужный столбец. Собирает единую таблицу.

        :param columns_mapping:- {string:string} -словарь с расшифровками колонок
        :return: array[string] - список возможных тестов
        '''
        self.df_result = pd.DataFrame(columns=self.headers)
        #получаем название столбца и путь, откуда его брать
        for col_name, file_col_list in columns_mapping.items():
            #делим путь до файла и название колонки в файлах пользователя
            for file_col in file_col_list:
                file_path, col_name_in_file = file_col.split("->")
                if not os.path.exists(file_path):
                    print(f"Предупреждение: Файл {file_path} не найден. Пропуск.")
                    continue

                if file_path.endswith(".xlsx") or file_path.endswith(".xls"):
                    data = pd.read_excel(file_path)
                elif file_path.endswith(".txt"):
                    data = pd.read_csv(file_path, delimiter="\t")
                else:
                    print(f"Предупреждение: Неизвестный формат файла {file_path}. Пропуск.")
                    continue

                self.df_result.loc[:, col_name] = data[col_name_in_file]
        #сортируем df так, чтобы все пустые колонки были справа
        column_order = np.concatenate(
            [self.df_result.columns[~self.df_result.isna().all(axis=0)], self.df_result.columns[self.df_result.isna().all(axis=0)]])
        self.df_result = self.df_result[column_order]
        columns_with_data = []

        # Проходим по DataFrame и добавляем названия колонок с данными в список
        for col in self.df_result.columns:
            if self.df_result[col].notna().any():
                columns_with_data.append(col)
        return self.get_possible_tests(columns_with_data)

    def start_tests(self, tests=None):
        '''
        Запускает выбранные тесты через QA_QC_kern
        :param tests: {string:[string]}
        :return: {string:{string:[int]}}
        '''
        test_array = set()
        if tests is not None:
            for test in tests.items():
                for t in test[-1]:
                    test_array.add(t)
        #в случае, если необходимы тесты связанные с напрвалением
        if self.df_result["Направление"].notna().any() and\
                self.df_result["Ск"].notna().any() and\
                self.df_result["Лабораторный номер"].notna().any() and\
                self.df_result["Открытая пористость по жидкости"].notna().any() and\
                self.df_result["Плотность абсолютно сухого образца"].notna().any():
            self.parallel_data_parsing()
        if self.df_result["Кровля интервала отбора"].notna().any() and self.df_result["Подошва интервала отбора"].notna().any():
            self.interval_data_parsing()
        test_system = QA_QC_kern(pas=np.array(self.df_result["Плотность абсолютно сухого образца"]),
                                 note=np.array(self.df_result["Примечание"]),
                                 kno=np.array(self.df_result["So"]),
                                 kp_plast=np.array(self.df_result["Открытая пористость в пластовых условиях"]),
                                 density=np.array(self.df_result["Плотность абсолютно сухого образца"]),
                                 water_permeability=np.array(self.df_result["Газопроницаемость по воде"]),
                                 kp_pov=np.array(self.df_result["Открытая пористость по жидкости"]),
                                 perpendicular=np.array(self.perpendicular_number),
                                 perpendicular_porosity=np.array(self.perpendicular_porosity),
                                 perpendicular_density=np.array(self.perpendicular_density),
                                 perpendicular_carbonate=np.array(self.perpendicular_carbonate),
                                 parallel=np.array(self.parallel_number),
                                 parallel_porosity=np.array(self.parallel_porosity),
                                 parallel_density=np.array(self.parallel_density),
                                 parallel_carbonate=np.array(self.parallel_carbonate),
                                 kp=np.array(self.df_result["Открытая пористость по жидкости"]),
                                 top=np.array(self.df_result["Кровля интервала отбора"]),
                                 core_removal_in_meters=np.array(self.df_result["Вынос керна, м"]),
                                 intervals=self.interval,
                                 bottom=np.array(self.df_result["Подошва интервала отбора"]),
                                 percent_core_removal=np.array(self.df_result["Вынос керна, %"]),
                                 outreach_in_meters=np.array(self.df_result["Вынос керна, м"]),
                                 sw_residual=np.array(self.df_result["Кво"]),
                                 core_sampling=np.array(self.df_result["Глубина отбора, м"]),
                                 kpr=np.array(self.df_result["Кпр_газ(гелий)"]),
                                 rp=np.array(self.df_result["Параметр пористости"]),
                                 pmu=np.array(self.df_result["Плотность максимально увлажненного образца"]),
                                 rn=np.array(self.df_result["Параметр насыщения"]),
                                 obplnas=np.array(self.df_result["Плотность абсолютно сухого образца"]),
                                 poroTBU=np.array(self.df_result["Открытая пористость в пластовых условиях"]),
                                 poroHe=np.array(self.df_result["Открытая пористость по газу"]),
                                 porosity_open=np.array(self.df_result["Открытая пористость по жидкости"]),
                                 porosity_kerosine=np.array(self.df_result["Открытая пористость по керосину"]),
                                 porosity_effective=np.array(self.df_result["Эффективная пористость"]),
                                 sw=np.array(self.df_result["Sw"]),
                                 parallel_permeability=np.array(self.df_result["Газопроницаемость, mkm2 (parallel)"]),
                                 klickenberg_permeability=np.array(self.df_result["Газопроницаемость по Кликенбергу"]),
                                 effective_permeability=np.array(self.df_result["Эффективная проницаемость"]),
                                 md=np.array(self.df_result["Место отбора (ниже кровли), м"]), show=False)

        self.failed_tests = test_system.start_tests(test_array)["wrong_parameters"]
        test_system.generate_test_report()
        self.error_flagging()
        return self.failed_tests

    def parallel_data_parsing(self):
        '''
        Делит данные на параллельные и перпендикулярные
        :return:
        '''
        direction_array = self.df_result["Направление"]
        for idx, is_parallel in enumerate(direction_array):
            if is_parallel == 1:
                self.parallel_density.append([self.df_result["Плотность абсолютно сухого образца"][idx], idx])
                self.parallel_porosity.append([self.df_result["Открытая пористость по жидкости"][idx], idx])
                self.parallel_number.append([self.df_result["Лабораторный номер"][idx], idx])
                self.parallel_carbonate.append([self.df_result["Ск"][idx], idx])
            else:
                self.perpendicular_density.append([self.df_result["Плотность абсолютно сухого образца"][idx], idx])
                self.perpendicular_porosity.append([self.df_result["Открытая пористость по жидкости"][idx], idx])
                self.perpendicular_number.append([self.df_result["Лабораторный номер"][idx], idx])
                self.perpendicular_carbonate.append([self.df_result["Ск"][idx], idx])

    def interval_data_parsing(self):
        '''
        Составляет массив из интервалов
        :return:
        '''
        top = self.df_result["Кровля интервала отбора"]
        bottom = self.df_result["Подошва интервала отбора"]
        for i in range(len(top)):
            self.interval.append([top[i], bottom[i]])

    def error_flagging(self):
        '''
        Красит в красный ячейки с ошибками
        :return:
        '''
        empty_series = pd.Series([])  # Создаем пустую Series
        self.df_result[""] = empty_series
        for test_name, failed_columns in self.failed_tests.items():
            # Создайте новый столбец в df_result для текущего названия теста
            self.df_result[test_name] = ""
            for column_data in failed_columns[:-1]:
                for col, indices in column_data.items():
                    # Найдите соответствующий столбец в df_result
                    if col in self.df_result.columns:
                        if col in self.newdic:
                            self.newdic[col].extend(indices)
                        else:
                            self.newdic[col]=indices
                        # Закрасьте ячейки в красный цвет, если индекс находится в массиве indices
                        for index in indices:
                            self.df_result.at[index, test_name] = failed_columns[-1]

    def color_cells(self, df, col_index_dict, color='red'):
        style = pd.DataFrame('', index=df.index, columns=df.columns)
        for col, rows in col_index_dict.items():
            for row in rows:
                style.loc[row, col] = f'background-color: {color}'
        return style

    def save_to_excel(self):
        styled_df = self.df_result.style.apply(self.color_cells, axis=None, col_index_dict=self.newdic, color='red')
        styled_df.to_excel("report\\post_test_table.xlsx", sheet_name='Sheet1', index=False)
