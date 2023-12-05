# Здесь реализуем классы для чтения всех требуемых форматов данных
import pandas as pd
import numpy as np
import lasio
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
                df_data = self.reading_smspec_unsmry(idx_spec = 0)
            elif f_extension == '.UNSMRY':
                df_data = self.reading_smspec_unsmry(idx_spec = 1)
            else:
                return 'Error format'
        else:
            _, f_extension = os.path.splitext(self.files)
            if f_extension == '.vol':
                df_data = self.reading_vol()
            else: 
                return 'Error format'

        return df_data
    
    def reading_smspec_unsmry(self, idx_spec = 0): 
        """ Функция  читает и обрабатывает бинарные файлы формата RSM

            Args:
                self.files (tuple): Имя файла, или кортеж имен файлов, которые нужно прочитать.\n
                self.files (int): индекс файла с расширением .SMSPEC \n
                
            Returns:
                df_data: pd.DataFrame, Датафрейм с узлами скважин, Индексы строк - дата, колонки - "имя_узла:имя_скважины'
                
                
        """
        sum = EclSum.load(os.path.join(self.data_folder,self.files[idx_spec]), 
                          os.path.join(self.data_folder,self.files[abs(idx_spec-1)]))
        column_keys = []
        for k in self.keywords_debit.keys():
            column_keys.append('W'+k+':*')
        for k in self.keywords_cumulative.keys():
            column_keys.append('W'+k+':*')
        for k in self.keywords_WEFAC.keys():
            column_keys.append(k+':*')
                        
        df_data = sum.pandas_frame(column_keys = column_keys)

        drop_list = []
        rename_col = {}
        for col in df_data.columns:
            if df_data[col].values.sum() == 0:
                drop_list.append(col)
            else:
                new_col = re.sub('[!@#$*]', '', col[1:]) 
                rename_col[col] = new_col

        df_data = df_data.drop(drop_list, axis =1)
        df_data.rename(columns = rename_col, inplace = True )
        
        return df_data
    
    def reading_vol(self):
        """ Функция  читает и обрабатывает файл с расширением .vol 

            Args:
                self.files (str): Имя файла, или кортеж имен файлов, которые нужно прочитать.\n
                                
            Returns:
                df_data: pd.DataFrame, Датафрейм с узлами скважин, Индексы строк - дата, колонки - "имя_узла:имя_скважины'
                
                
        """
        with open(os.path.join(self.data_folder,self.files), 'r') as f:
            lines = f.readlines()

            lines = list(map(str.strip, lines))
            lines = list(filter(None, lines))
            idx_well_name_lines = [n for n, line in enumerate(lines) if line.startswith('*NAME')] 

            columns_file = lines[idx_well_name_lines[0]-1].replace('*', '').split()
            words_header = [line.replace('\t', ' ').split()[0] for line in lines[:idx_well_name_lines[0]-1]]
            
            if '*CUMULATIVE' in words_header:
                keywords = self.keywords_cumulative|self.keywords_WEFAC
            else:
                keywords = self.keywords_debit|self.keywords_WEFAC                

            idx_keywords = dict()
            for k, v in keywords.items():
                if v in columns_file:
                    idx_keywords[columns_file.index(v)] = k
            
            data = []
            dates  = []
            columns_df = []
            idx_col_sort = list(idx_keywords.keys())
            idx_col_sort.sort()
            for i in range(len(idx_well_name_lines)):
                well_name = lines[idx_well_name_lines[i]].replace('\t', ' ').split()[1]
                well_name = re.sub('[!@#$*]', '', well_name)
                
                for idx in idx_col_sort:
                    columns_df.append(idx_keywords[idx]+':'+well_name)

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

                if i == len(idx_well_name_lines)-1:
                    end_idx = len(lines)
                else:
                    end_idx = idx_well_name_lines[i+1] 

                for line in lines[idx_well_name_lines[i]+1:end_idx]:
                    line_list = line.replace('\t', ' ').split()

                    if read_dates:
                        dates.append(datetime.datetime(int(line_list[idx_Y]), int(line_list[idx_M]), int(line_list[idx_D]), 
                                                       int(line_list[idx_h]), int(line_list[idx_m]), int(line_list[idx_s])))
                        
                    data_well.append([line_list[idx] for idx in idx_col_sort])

                data.append(np.array(data_well))

        data = np.concatenate(data, axis = 1).astype(float)
        df_data = pd.DataFrame(index=dates, columns = columns_df, data=data)
        drop_list = []
        for col in columns_df:
            if df_data[col].values.sum() == 0:
                drop_list.append(col)

        df_data = df_data.drop(drop_list, axis =1)

        return df_data
    

class Reader_gis_data_for_well:
    def __init__(self, name_stratum: str, data_folder: str, mnemonics_file_name: str, tops_formation_file_name: str):
        """ Читает входные файлы с историческими данными по скважинам.

            Args:
                name_stratum (str): имя пласта.\n
                data_folder (str): Путь до папки, где лежат файлы\n
                mnemonics_file_name (str): Имя exel файла с мнемониками узлов ГИС.\n
                tops_formation_file_name (str): Имя exel файла c кровлей и подошвой пласта в точках скважины.\n
        """

        self.name_stratum = name_stratum
        self.mnemonics_file_name = mnemonics_file_name
        self.tops_formation_file_name = tops_formation_file_name
        self.top_bottom_for_wells = self.read_top_bottom_stratum_for_wells(data_folder)
        self.mnemonics = self.read_mnemonics(data_folder)

    def read_mnemonics(self, data_folder: str):
        """ Считывает данные о мнемониках узлов ГИС и формирует словарь мнемоник. \n

            Args:
                data_folder (str): Путь до папки, где лежат файлы\n
                self.mnemonics_file_name (str): Имя exel файла с мнемониками узлов ГИС.\n
            Returns:
                mnemonics: dict, ключи - имена узла, значения - список мнемоник.
            
        """
        df_mnemonics = pd.read_excel(os.path.join(data_folder,self.mnemonics_file_name), index_col=0)
        mnemonics = df_mnemonics['Мнемоники'].to_dict()
        for k, v in mnemonics.items():
            v = v.split(',')
            mnemonics[k] = [m.strip() for m in v if len(m.strip())!=0]

        return mnemonics

    def read_top_bottom_stratum_for_wells(self, data_folder: str):
        """ Считывает данные о кровле и подошве пласта в точках скважин и формирует словарь по скважинам. \n

            Args:
                data_folder (str): Путь до папки, где лежат файлы\n
                self.tops_formation_file_name (str): Имя exel файла c кровлей и подошвой пласта в точках скважины.\n
                self.name_stratum (str): имя пласта.\n
            Returns:
                top_bottom_for_wells: dict, ключи - имена скважин, значения - кортеж глубин (кровля, подошва) в метрах.
            
        """
        df = pd.read_excel(os.path.join(data_folder,self.tops_formation_file_name))
        df = df[df['Surface'].str.contains(self.name_stratum)]
        top_bottom_for_wells = {}
        for w in df['Well identifier']:
            top = None
            bottom = None
            w_df = df[df['Well identifier'] == w]
            if len(w_df) !=0:
                if len(w_df[w_df['Surface'].str.contains('top')]) !=0:
                    top = w_df[w_df['Surface'].str.contains('top')]['MD'].values[0]
                if len(w_df[w_df['Surface'].str.contains('bot')]) !=0:
                    bottom = w_df[w_df['Surface'].str.contains('bot')]['MD'].values[0]
            if top:
                top_bottom_for_wells[w] = (top,bottom)

        return top_bottom_for_wells
                        
    def reading_gis_data(self, data_folder: str, las_file_name: str):
        """ Функция читает и обрабатывает входныой las файл\n
            Args:
                data_folder (str): Путь до папки, где лежат файлы\n
                las_file_name (str): Имя файла, который нужно прочитать.\n
                
            Returns:
                well_name (str): имя скважины\n
                df_data: pd.DataFrame, Датафрейм c каротажами скважины\n       
        """
        las = lasio.read(os.path.join(data_folder, las_file_name))
        well_name = las.well.WELL.value.replace('Copy of ','')

        if not well_name in self.top_bottom_for_wells.keys():
            return 'Для скважины '+ well_name + ' нет данных о пластопересечении'
        else:
            top, bottom = self.top_bottom_for_wells[well_name]
        
        df_las = las.df()
        df_las = df_las[df_las.index > top]
        if bottom:
            df_las = df_las[df_las.index < bottom]

        df_las=df_las.dropna(axis=1,how='all')

        if len(df_las.columns) == 0:
            return well_name, 'Не найдены записи каротажей в пределах плата '+ self.name_stratum
 
        return well_name, df_las       
        
          