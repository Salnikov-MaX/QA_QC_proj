# Здесь реализуем классы для чтения всех требуемых форматов данных
import pandas as pd
import numpy as np
from ecl.summary import EclSum 
import os
import datetime
import re

class Reader_histor_well_data:
    def __init__(self, data_folder: str, files: tuple or str, keywords: dict):
        """
            Читает входные файлы с историческими данными по скважинам.

        Args:
            data_folder (str): Путь до папки, где лежат файлы\n
            files (tuple or st): Имя файла, или кортеж имен файлов, которые нужно прочитать.\n
        """

        self.data_folder = data_folder
        self.files = files
        self.keywords_debit = keywords[0]
        self.keywords_cumulative = keywords[1]
        self.keywords_WEFAC = keywords[2]

    def reading_wells_data(self):
        """ Функция в зависимости от формата читает и обрабатывает входные файлы

            Args:
                self.files (tuple or st): Имя файла, или кортеж имен файлов, которые нужно прочитать.\n
                
            Returns:
                df_data: pd.DataFrame, Датафрейм с узлами скважин, Индексы строк - дата, колонки - "имя_узла:имя_скважины'
                
                
        """

        if type(self.files) is tuple:
            _, f_extension = os.path.splitext(self.files[0])
            
            if f_extension == '.SMSPEC':
                df_data = self.reading_smspec_unsmry(idx_spec=0)
            elif f_extension == '.UNSMRY':
                df_data = self.reading_smspec_unsmry(idx_spec=1)
            else:
                return 'Error format'
        else:
            _, f_extension = os.path.splitext(self.files)
            if f_extension == '.vol':
                df_data = self.reading_vol()
            else:
                return 'Error format'

        return df_data

    def reading_smspec_unsmry(self, idx_spec=0):
        """ Функция  читает и обрабатывает бинарные файлы формата RSM

            Args:
                self.files (tuple): Имя файла, или кортеж имен файлов, которые нужно прочитать.\n
                self.files (int): индекс файла с расширением .SMSPEC \n
                
            Returns:
                df_data: pd.DataFrame, Датафрейм с узлами скважин, Индексы строк - дата, колонки - "имя_узла:имя_скважины'
                
                
        """
        sum = EclSum.load(os.path.join(self.data_folder, self.files[idx_spec]),
                          os.path.join(self.data_folder, self.files[abs(idx_spec - 1)]))
        column_keys = []
        for k in self.keywords_debit.keys():
            column_keys.append('W' + k + ':*')
        for k in self.keywords_cumulative.keys():
            column_keys.append('W' + k + ':*')
        for k in self.keywords_WEFAC.keys():
            column_keys.append(k + ':*')

        df_data = sum.pandas_frame(column_keys=column_keys)

        drop_list = []
        rename_col = {}
        for col in df_data.columns:
            if df_data[col].values.sum() == 0:
                drop_list.append(col)
            else:
                new_col = re.sub('[!@#$*]', '', col[1:])
                rename_col[col] = new_col

        df_data = df_data.drop(drop_list, axis=1)
        df_data.rename(columns=rename_col, inplace=True)

        return df_data

    def reading_vol(self):
        """ Функция  читает и обрабатывает файл с расширением .vol 

            Args:
                self.files (str): Имя файла, или кортеж имен файлов, которые нужно прочитать.\n
                                
            Returns:
                df_data: pd.DataFrame, Датафрейм с узлами скважин, Индексы строк - дата, колонки - "имя_узла:имя_скважины'
                
                
        """
        with open(os.path.join(self.data_folder, self.files), 'r') as f:
            lines = f.readlines()

            lines = list(map(str.strip, lines))
            lines = list(filter(None, lines))
            idx_well_name_lines = [n for n, line in enumerate(lines) if line.startswith('*NAME')]

            columns_file = lines[idx_well_name_lines[0] - 1].replace('*', '').split()
            words_header = [line.replace('\t', ' ').split()[0] for line in lines[:idx_well_name_lines[0] - 1]]

            if '*CUMULATIVE' in words_header:
                keywords = self.keywords_cumulative | self.keywords_WEFAC
            else:
                keywords = self.keywords_debit | self.keywords_WEFAC

            idx_keywords = dict()
            for k, v in keywords.items():
                if v in columns_file:
                    idx_keywords[columns_file.index(v)] = k

            data = []
            dates = []
            columns_df = []
            idx_col_sort = list(idx_keywords.keys())
            idx_col_sort.sort()
            for i in range(len(idx_well_name_lines)):
                well_name = lines[idx_well_name_lines[i]].replace('\t', ' ').split()[1]
                well_name = re.sub('[!@#$*]', '', well_name)

                for idx in idx_col_sort:
                    columns_df.append(idx_keywords[idx] + ':' + well_name)

                if i == 0:
                    read_dates = True
                    idx_Y = columns_file.index('YEAR')
                    idx_M = columns_file.index('MONTH')
                    idx_D = columns_file.index('DAY')
                    idx_h = columns_file.index('HOUR')
                    idx_m = columns_file.index('MINUTE')
                    idx_s = columns_file.index('SECOND')
                else:
                    read_dates = False

                data_well = []

                if i == len(idx_well_name_lines) - 1:
                    end_idx = len(lines)
                else:
                    end_idx = idx_well_name_lines[i + 1]

                for line in lines[idx_well_name_lines[i] + 1:end_idx]:
                    line_list = line.replace('\t', ' ').split()

                    if read_dates:
                        dates.append(
                            datetime.datetime(int(line_list[idx_Y]), int(line_list[idx_M]), int(line_list[idx_D]),
                                              int(line_list[idx_h]), int(line_list[idx_m]), int(line_list[idx_s])))

                    data_well.append([line_list[idx] for idx in idx_col_sort])

                data.append(np.array(data_well))

        data = np.concatenate(data, axis=1).astype(float)
        df_data = pd.DataFrame(index=dates, columns=columns_df, data=data)
        drop_list = []
        for col in columns_df:
            if df_data[col].values.sum() == 0:
                drop_list.append(col)

        df_data = df_data.drop(drop_list, axis=1)

        return df_data
    

   
        
          
