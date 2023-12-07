import pandas as pd
import os
import lasio

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