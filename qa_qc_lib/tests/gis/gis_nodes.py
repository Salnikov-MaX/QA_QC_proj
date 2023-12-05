import pandas as pd
import numpy as np
from qa_qc_lib.readers.gis_reader import Reader_gis_data_for_well

class Nodes_gis_data:
    def __init__(self, data_folder: str, las_files_name: str, reader_gis_data: Reader_gis_data_for_well):
        """ Класс содержащий узлы типа скв.иссл и сопутствующие атрибуты

            Args:
                data_folder (str): Путь до папки, где лежат файлы\n
                las_files_name (str): Имя las файла с каротажами ГИС/РИГИС \n
                reader_gis_data (Reader_gis_data_for_well): ридер для парсинга las файла\n
        """

        self.data_folder = data_folder
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
        self.well_name, df_las = self.reader.reading_gis_data(self.data_folder, self.las_files_name)
        if type(df_las) == str:
            print('Невозможно обработать файл ', self.las_files_name)
            print(df_las)
            raise Exception()
        
        self.dept = df_las.index.values
        self.top, self.bottom = self.reader.top_bottom_for_wells[self.well_name] 
        
        self.gis_nodes = dict()
        las_columns = df_las.columns
        self.find_mnems = dict()
        for node_type in self.reader.mnemonics.keys():
            nodes = dict()
            mnems = self.reader.mnemonics[node_type]
            for mn in mnems:
                if mn[-1]== '*':
                    col_idx = np.where(las_columns.str.contains('^'+mn[:-1], regex=True))[0]
                else:
                    col_idx = np.where(las_columns == mn)[0]
                if len(col_idx) > 0:
                    for idx in col_idx:
                        if las_columns[idx] not in self.find_mnems.keys():
                            self.find_mnems[las_columns[idx]] = {node_type}
                        else:
                            self.find_mnems[las_columns[idx]].add(node_type)
                            
                        nodes[las_columns[idx]] = df_las[las_columns[idx]].values 
            if len(nodes) !=0:
                self.gis_nodes[node_type] = nodes
        self.gis_nodes['Ignor'] = set(las_columns.to_list()) - set(self.find_mnems.keys())

    def check_data(self):
        """ Проверят обработанные данные и выводит информацию для проверки пользователем \n
            Args:
                self.gis_nodes (dict): словарь типов узлов и принадлежащих им каротажей.\n
                self.find_mnems (dict): словарь с именами кратажей и типами узлов, к которым они относятся.\n
                                            
        """
        print('Проверка данных файла '+ self.las_files_name)
        print('1. Проверка на неоднозначность имен каротажей.')
        i = 0
        for k, v in self.find_mnems.items():
            if len(v) >1:
                i+=1
                print('---Имя каротажа '+ k + ' и используется в узлах:', v)
        if i == 0:
            print('---неоднозначности отсутствуют!')

        print()
        print('2. Игнорируемые каротажи:')
        if len(self.gis_nodes['Ignor']) != 0:
            print('---Для следующих каротажей не удалось определить узел и они будут проигнорированы при тестировании:')
            print(self.gis_nodes['Ignor'])
            print('---Уточните файл с мнемониками')
        else:
            print('---в данных нет игнорируемых каротажей')
        
        print()
        print('3. Наличие каротажей для тестирования.')
        if len(self.gis_nodes.keys()) == 1:
            print('---Не обнаружены каротажи с записями для тестирования в пределах пласта '+ self.reader.name_stratum)
        else:
            for k,v in self.gis_nodes.items():
                if k != 'Ignor':
                    print('---В узеле '+ k+ ' каротажи: ', list(v.keys()))    
                  

