import pandas as pd
import numpy as np
from qa_qc_lib.readers.well_reader import Reader_histor_well_data


class Nodes_wells_data:
    def __init__(self, data_folder: str, histor_files_names: tuple or str, wpi_file_name: str = None,
                 pgi_file_name: str = None):
        """Класс содержащий узлы типа скв.иссл и сопутствующие атрибуты

        Args:
            data_folder (str): Путь до папки, где лежат файлы\n
            histor_files_names (tuple or str): Имя файла, или кортеж имен файлов с историческими данными по скважинам.Defaults to None\n
            wpi_file_name (str): wpi (well perforation intervals) имя файла с интевалами перфораций скважин. Defaults to None \n
            keywords_debit (dict): словарь с ключевыми словами имен узлов дебета и соответствующими им имена колонок во входных файлах\n
            keywords_cumulative (dict): словарь с ключевыми словами имен узлов накопительных данных и \n
                                   соответствующими им имена колонок во входных файлах\n
            keywords_wefac (dict): словарь с ключевыми словами имен узлов коэффициента эксплуатации скважины\n
           
        """

        self.data_folder = data_folder
        self.histor_files_names = histor_files_names
        self.wpi_file_name = wpi_file_name
        self.pgi_file_name = pgi_file_name

        self.keywords_debit = {'OPR': 'OIL', 
                                'WPR': 'WATER',
                                'GPR': 'GAS',
                                'LPR': 'WLPR',
                                'BHP': 'BHP',
                                'WIR': 'WINJ',
                                'GIR': 'GINJ'}

        self.keywords_cumulative = {'OPT': 'OIL',
                                    'WPT': 'WATER',
                                    'LPT': 'WLPT',
                                    'GPT': 'GAS',
                                    'WIT': 'WINJ', 
                                    'GIT': 'GINJ'}
        
        self.keywords_wefac = {'WEFAC': 'UPTIME'}
        
        self.init_data()        
        
    def init_data(self):

        """ Функция считывает входные файлы и заполняет первоночальные данные при инициализации класса \n

            Args:
                self.nodes_wells (dict): Словарь узлов скважин \n
                self.wells (list): список скважин \n
                self.time_scale (np.array): временная шкала входных данных \n
                self.init_idx_ts_wells (dict): хранит индексы начала работы скважины \n
                            
        """

        df_histor_data = Reader_histor_well_data(self.data_folder, self.histor_files_names, 
                                                [self.keywords_debit,self.keywords_cumulative,self.keywords_wefac]).reading_wells_data()
        if type(df_histor_data) == str:
            print('Невозможно прочитать файл ', self.histor_files_names)
            print(df_histor_data)
            raise Exception()
        
        if self.wpi_file_name == None:
            df_wpi_data = None

        if self.pgi_file_name == None:
            df_pgi_data = None

        self.wells = self.get_wells(df_histor_data, df_wpi_data, df_pgi_data)
        self.time_scale = df_histor_data.index.date 
        self.nodes_wells = self.get_nodes(df_histor_data, df_wpi_data, df_pgi_data)
        self.init_idx_ts_wells = self.get_idx_begin_work_wells()
        
    def get_idx_begin_work_wells(self):
         
        """ Функция ищет индекс первого ненулевого значения во всех исторических данных (временных рядах).\n
            Для каждой скважины берет один минимальный индекс из всех временных рядов

            Args:
                self.nodes_wells (dict): Словарь узлов скважин \n
                self.wells (list): список скважин \n
            Returns:
                idxs_begin_work_wells: dict, ключи - имена скважин, значения - начальные индексы.
                {well_name: init_idx}
                
        """
        idxs_begin_work_wells = dict()
        keys_nodes_histor_data = set((self.keywords_debit|self.keywords_cumulative).keys())

        for w in self.wells:
            init_idxs = list()            
            nodes_w = self.nodes_wells[w] 
            histor_nodes_w = set(nodes_w.keys())&keys_nodes_histor_data
            for k in histor_nodes_w:
                idx_mask = nodes_w[k] !=0
                init_idxs.append(idx_mask.argmax())

            if len(init_idxs)!=0:
                idxs_begin_work_wells[w] = min(init_idxs)
            
        return idxs_begin_work_wells
    
    def get_wells(self, df_histor_data, df_wpi_data, df_pgi_data):

        """ Функция выбирает из всех загруженных данных имена скважин

            Args:
                df_histor_data (pd.DataFrame): датасет c историческими данными по скважинам \n
                df_wpi_data (pd.DataFrame): датасет с интервалами перфораций по скважинам \n
                df_pgi_data (pd.DataFrame): датасет с отчетом ПГИ. \n 
            Returns:
                wells (list): список имен скважин.
        """

        wells = [col.split(':')[-1] for col in df_histor_data.columns]
        if df_wpi_data != None:
            wells+= [col for col in df_wpi_data.columns]

        if df_pgi_data != None:
            wells += [col for col in df_pgi_data.columns]

        return list(set(wells))

    def get_nodes(self, df_histor_data, df_wpi_data, df_pgi_data):
        """Функция создает структуру данных с узлами каждой скважины

            Args:
                df_histor_well_data (pd.DataFrame): исходный датасет c историческими данными по скважинам \n
                df_wpi_data (pd.DataFrame): датасет с интервалами перфораций по скважинам \n
                df_pgi_data (pd.DataFrame): датасет с отчетом ПГИ. \n
                self.wells (list): спислк имен скважин

            Returns:
                w_nodes: dict, словарь с узлами скважин.
                {well_name (str): {name_node (str): value (np.array)}}
        """
        w_nodes = dict()
        for w in self.wells:
            nodes = dict()
            for col in df_histor_data.columns:
                pars_col = col.split(':')
                if pars_col[1] == w:
                    nodes[pars_col[0]] = df_histor_data[col].values
            if df_wpi_data != None:
                continue
            if df_pgi_data != None:
                continue
            w_nodes[w] = nodes

        return w_nodes
    