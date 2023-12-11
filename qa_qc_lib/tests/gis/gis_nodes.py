import pandas as pd
import numpy as np
from qa_qc_lib.readers.gis_reader import Reader_gis_data_for_well


class Nodes_gis_data:
    def __init__(self, las_files_name: str,
                 reader_gis_data: Reader_gis_data_for_well):
        """ Класс содержащий узлы типа скв.иссл и сопутствующие атрибуты

            Args:
                las_files_name (str): Имя las файла с каротажами ГИС/РИГИС \n
                reader_gis_data (Reader_gis_data_for_well): ридер для парсинга las файла\n
        """

        self.las_files_name = las_files_name
        self.reader = reader_gis_data
        self.init_data()

    def init_data(self):
        """ Первоначальное заполнение атрибутов класса\n
            Args:
                self.raeder (Reader_gis_data_for_well): ридер для парсинга las файла\n
                self.well_name (str): имя скважины из las файла\n
                self.dept (np.array): массив глубин.\n
                self.top (float): кровля пласта.\n
                self.bottom (float): подошва пласта.\n 
                self.gis_nodes (dict): словарь типов узлов и принадлежащих им каротажей.\n
                self.find_mnems (dict): словарь с именами кратажей и типами узлов, к которым они относятся.\n
            Returns:
                None

        """
        self.well_name, df_las = self.reader.reading_gis_data(self.las_files_name)

        self.dept = df_las.index.values
        self.top, self.bottom = self.reader.top_bottom_for_wells[self.well_name]

        self.gis_nodes = dict()
        las_columns = df_las.columns
        self.find_mnems = dict()
        for node_type in self.reader.mnemonics.keys():
            nodes = dict()
            mnems = self.reader.mnemonics[node_type]
            for mn in mnems:
                if mn[-1] == '*':
                    col_idx = np.where(las_columns.str.contains('^' + mn[:-1], regex=True))[0]
                else:
                    col_idx = np.where(las_columns == mn)[0]
                if len(col_idx) > 0:
                    for idx in col_idx:
                        if las_columns[idx] not in self.find_mnems.keys():
                            self.find_mnems[las_columns[idx]] = {node_type}
                        else:
                            self.find_mnems[las_columns[idx]].add(node_type)

                        nodes[las_columns[idx]] = df_las[las_columns[idx]].values
            if len(nodes) != 0:
                self.gis_nodes[node_type] = nodes
        self.gis_nodes['Ignor'] = set(las_columns.to_list()) - set(self.find_mnems.keys())

    def check_data(self) -> str:
        """ Проверят обработанные данные и выводит информацию для проверки пользователем \n
            Args:
                self.gis_nodes (dict): словарь типов узлов и принадлежащих им каротажей.\n
                self.find_mnems (dict): словарь с именами кратажей и типами узлов, к которым они относятся.\n

            Returns:
                report (str): Общий отчёт о качестве входных данных
                                            
        """
        report = ['Проверка данных файла ' + self.las_files_name,
                  '1. Проверка на неоднозначность имен каротажей.',
                  '\n']
        i = 0
        for k, v in self.find_mnems.items():
            if len(v) > 1:
                i += 1
                report.append('---Имя каротажа ' + k + ' и используется в узлах:', v)
        if i == 0:
            report.append('---неоднозначности отсутствуют!')

        report.append('\n')
        report.append('2. Игнорируемые каротажи:')

        report.append('2. Игнорируемые каротажи:')
        if len(self.gis_nodes['Ignor']) != 0:
            report.append(
                '---Для следующих каротажей не удалось определить узел и они будут проигнорированы при тестировании:')
            report.append(', '.join(self.gis_nodes['Ignor']))
            report.append('---Уточните файл с мнемониками')
        else:
            report.append('---в данных нет игнорируемых каротажей')

        report.append('\n')
        report.append('3. Наличие каротажей для тестирования.')
        if len(self.gis_nodes.keys()) == 1:
            report.append(
                '---Не обнаружены каротажи с записями для тестирования в пределах пласта ' + self.reader.name_stratum)
        else:
            for k, v in self.gis_nodes.items():
                if k != 'Ignor':
                    report.append('---В узеле ' + k + ' каротажи: ')
                    report.append(', '.join(list(v.keys())))

        # print("\n".join(report))
        return "\n".join(report)
