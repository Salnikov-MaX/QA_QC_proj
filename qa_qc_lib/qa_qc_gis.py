import lasio
import pandas as pd
import numpy as np
import datetime
import re
from .qa_qc_main import QA_QC_main
from .qa_qc_tools.gis_tools import *


class QA_QC_gis(QA_QC_main):
    def __init__(self, las_path:str, well_tops_path:str = None) -> None:

        super().__init__()
        las = lasio.read(las_path)
        self.file_name = las_path.split('/')[-1]
        self.las_df = las.df()
        self.units_dict = {curve.mnemonic.upper() : curve.unit for curve in las.curves}

        if well_tops_path: 
            self.well_tops_df = pd.read_excel(well_tops_path)  # структура "Surface", "X", "Y", "Z", "MD" | в файле должны быть данные только по данной скважине
            income_list = [name in self.well_tops_df.columns for name in ["Surface", "X", "Y", "Z", "MD"]]
            assert all(income_list), 'В файле отсутствуют ожидаемые колонки ("Surface", "X", "Y", "Z", "MD" ), убедитесь в корректности файла'

        self.__mnemonics = {'SP'   : ['SP', 'PS', 'ПС', 'СП', 'PS_1', 'PS_2'],
                            'GR'   : ['GR', 'GK', 'ГК', 'ECGR', 'GK_1', 'GK_2'],
                            'DS'   : ['DS', 'CALI', 'HCAL', 'CALIP', 'DCAV', 'DSN', 'DS_1', 'DS2'],
                            'MDS'  : ['MCAL', 'MDS'],
                            'MINV' : ['МГЗ', 'MGZ'],
                            'MNOR' : ['МПЗ', 'MPZ'],
                            'MLL'  : ['MLL', 'MBK', 'МБК', 'МКЗ', 'MSFL', 'RXOZ'],
                            'RHOB' : ['RHOB', 'PL', 'GGKP', 'RHOZ', 'DRHB', 'ROBB'],
                            'PEF'  : ['PEF', 'PE', 'PEFZ', 'ZEFF'],
                            'DT'   : ['AK', 'DT', 'DTp', 'DTP', 'АК', 'AK_2', 'DTL', 'DTS', 'DTP1'],
                            'ILD'  : ['IK', 'ILD', 'ИК', 'CILD', 'IKA', 'ILDA', 'IK_1', 'R27PC_46PH'],
                            'ILDR' : ['IKR', 'ILDR'],
                            'VIKIZ': ['ВИКИЗ', 'VIKIZ', 'F05', 'F07', 'F10', 'F14', 'F20', 'R05', 'R07', 'R10', 'R14', 'R20', 'С05', 'С07', 'С10', 'С14', 'С20', 'AT10', 'AT20', 'AT30', 'AT60', 'AT90', 'AF10', 'AF20', 'AF30', 'AF60',
                                      'AF90', 'ILDVG1', 'ILDVG2', 'ILDVG3', 'ILDVG4', 'ILDVG5', 'ILDVR1', 'ILDVR2', 'ILDVR3', 'ILDVR4', 'ILDVR5', 'R1', 'R2', 'R3', 'R4', 'R5', 'VIK1', 'VIK2', 'VIK3', 'VIK4', 'VIK5', 'RO05', 'RO07',
                                      'RO10', 'RO14', 'RO20', 'IK1', 'IK2', 'IK3', 'IK4', 'IK5', 'С07', 'С10', 'С14', 'С20', 'AT10', 'AT20', 'AT30', 'AT60', 'AT90', 'AF10', 'AF20', 'AF30', 'AF60', 'AF90', 'ILDVG1', 'ILDVG2',
                                      'ILDVG3', 'ILDVG4', 'ILDVG5', 'ILDVR1', 'ILDVR2', 'ILDVR3', 'ILDVR4', 'ILDVR5', 'R1', 'R2', 'R3', 'R4', 'R5', 'VIK1', 'VIK2', 'VIK3', 'VIK4', 'VIK5', 'RO05', 'RO07', 'RO10', 'RO14', 'RO20',
                                      'IK1', 'IK2', 'IK3', 'IK4', 'IK5'],
                            'NGR'  : ['NEUT', 'NGR', 'NGK', 'NGK_1'],
                            'NKTD' : ['NKTD', 'CFTC', 'NKT', 'NKTB', 'NKT_1', 'NKTB2'],
                            'NKTS' : ['NKTS', 'CNTC', 'NKTM', 'NKT_2'],
                            'W'    : ['NPHI', 'W', 'TNPH', 'NPLS', 'NPSS', 'TNPD', 'TNPL', 'TNPS', 'TNPH', 'TNPH_DOL', 'TNPH_LIM', 'TNPH_SAN'],
                            'DEPTH': ['DEPT']
                            }

        self.check_gis_names()


    def return_mnemonics(self) -> dict:
        """
        Метод возвращает копию словаря с мнемониками. При изменении данной копии, __mnemonics определенная в библиотеке не изменится.
        Для расширения списка мнемоник используйте метод add_mnemonics.

        Returns:
            dict: копия словаря с мнемониками
        """        
        return self.__mnemonics.copy()


    def add_mnemonics(self, key:str, values:list):
        """
        Метод для расширения словаря мнемоник.

        Args:
            key (str): ключ - стандартное название каротажа
            values (list): альтернативные названия каротажа key 
        """        
        key = key.upper()
        self.__mnemonics[key] = self.__mnemonics.get(key, []) + values


    def remove_mnemonics(self, key:str, values_to_remove:list):
        """
        Метод для удаления определенных значений мнемоник из списка, связанного с указанным ключом.

        Args:
            key (str): ключ - стандартное название каротажа
            values_to_remove (list): список значений мнемоник, которые нужно удалить из списка, связанного с ключом
        """
        key = key.upper()
        # Проверяем, есть ли ключ в словаре с помощью assert
        assert key in self.__mnemonics, f"Key '{key}' not found in mnemonics dictionary"

        # Удаляем каждое значение из списка, связанного с ключом
        for value in values_to_remove:
            if value in self.__mnemonics[key]:
                self.__mnemonics[key].remove(value)


    def find_mnemonic(self, search_value) -> str:
        """
        Метод для поиска стандартного названия ГИС из словаря мнемоник

        Args:
            search_value (_type_): название каротажа которому нужно найти соответствие
        Returns:
            _type_: стандартное название ГИС соответствующий search_value
        """        
        for key, values in self.__mnemonics.items():
            if search_value in values:
                return key.upper()
        return None


    def check_gis_names(self):
        """
        Метод опознает ГИС из файла las по их названиям и расширяет ими словарь мнемоник; 
        создаёт список неопознанных ГИС (unidentified_gis) находящихся в переданом las. 
        Словарь мнемоник является приватным, для взаимодействия с ним используйте следующие методы: 
        return_mnemonics, add_mnemonics, remove_mnemonics, find_mnemonic
        """        
        self.unidentified_gis = []
        for gis in self.las_df.columns:
            gis = gis.upper()
            if not self.find_mnemonic(gis):
                gis_mnem = re.sub(r'(-_\d+)?(-\d+)?', '', gis)
                mnem_key = self.find_mnemonic(gis_mnem)
                if mnem_key:
                    self.add_mnemonics(mnem_key, [gis])
                    print(f'Удалось найти соответствие в мнемониках: "{gis} - {mnem_key}", словарь мнемоник расширен')
                else:
                    self.unidentified_gis.append(gis)
        if self.unidentified_gis: print(f"""ВНИМАНИЕ! Не удалось распознать следующие каротажи: {self.unidentified_gis}

          Для того чтобы эти каротажи учавствовали в дальнейших тестах, следует расширить словарь мненмоник (используйте метод add_mnemonics), 
          а затем снова провести проверку, используя метод check_gis_names, чтобы список unidentified_gis сформировался заново.""")


    def test_physical_correctness(self, get_report=True) -> dict:
        """
        Метод для оценки физичности значений кривых ГИС
        Внимание! Тест не работает с неопознанными ГИС (self.unidentified_gis)

        Required data:
            self.las_df (Pandas.DataFrame): датафрейм содержащий кривые ГИС скважины

        Args:
            get_report (bool, optional): Определяет, нужно ли отображать отчет. Defaults to True.

        Returns:
            dict: Словарь с ключевыми данными о прохождении теста
        """        
        # Словарь содержащий условия прохождения теста для каждого из видов ГИС
        conditions_dict = {
            'SP'   : lambda x, multiplier: -1500 <= x*multiplier <= 1500 ,
            'DS'   : lambda x, multiplier: 0.1*multiplier <= x <= 0.5*multiplier,
            'MDS'  : lambda x, multiplier: 0.1*multiplier <= x <= 0.5*multiplier,
            'RHOB' : lambda x, multiplier: 1.5*multiplier <= x <= 3.5*multiplier,
            'PEF'  : lambda x, multiplier: 0.0*multiplier <= x <= 10.0*multiplier,
            'DT'   : lambda x, multiplier: 100.0*multiplier <= x <= 800.0*multiplier,
            'W'    : lambda x, multiplier: -0.15*multiplier <= x <= 1.0*multiplier,       
        }
        # Словарь содержащий множетель для перевода значений в соответствующую единицу измерения
        multiplier_dict = {
            'SP'   : {'mV':1, 'mv':1, 'V':0.001, 'v':0.001},
            'DS'   : {'cm':100, 'cm':100, 'm':1, 'м':1, 'mm':1000, 'мм':1000},
            'MDS'  : {'cm':100, 'cm':100, 'm':1, 'м':1, 'mm':1000, 'мм':1000},
            'RHOB' : {'g/cc':1, 'g/cm3':1, 'kg/m3':1000, 'кг/м3':1000},
            'PEF'  : {'b/e':1},
            'DT'   : {'us/m':1, 'us/f':0.3},
            'W'    : {'v/v':1, '%':100},
        }

        all_results_dict = {}

        for gis in self.las_df.columns:
            if gis not in self.unidentified_gis: # исключаем проведение тестирования для неопознанных ГИС
                results_dict = {}
                gis = gis.upper()
                key_gis = self.find_mnemonic(gis) # определяем типовое название для каротажа
                mult_dict = multiplier_dict.get(key_gis, 'another gis') # получаем множитель, в случае если ГИС не предусматривался получаем str 'another gis'
                try:
                    multiplier = mult_dict[self.units_dict[gis]] if mult_dict != 'another gis' else 1
                    # Непосредственно проведение проверки на физичность значений
                    test_result_mask = self.las_df[gis].apply(conditions_dict.get(gis, lambda x, y: x > 0), args=(multiplier,))
                    results_dict['test_result_mask'] = test_result_mask.to_list()
                    result = test_result_mask.all()
                    results_dict['result'] = str(result)
                    if result:
                        text = f'Значения коротажа {gis} физически корректны'
                        report_text = self.generate_report_text(text, 1)
                    else:
                        text = f'Значения коротажа {gis} физически не корректны'
                        report_text = self.generate_report_text(text, 0)
                    
                    # Добавим предупреждение о том, что данные проверялись только на условие "> 0", что может оказаться не всегда корректно
                    if mult_dict == 'another gis':
                        report_text += ' (Внимание, для этого ГИС проводилась проверка только на положительность значений)'
                
                except KeyError:
                    results_dict['result'] = 'Fail'
                    text = f'Неизвестная размерность данных {self.units_dict[gis]} для {gis}'
                    report_text = self.generate_report_text(text, 2)

                # Формируем отчет
                all_results_dict[gis] = results_dict
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.report_text += f"{timestamp:10} / test_physical_correctness:\n{report_text}\n\n"
                if get_report: print('\n'+report_text+self.delimeter)
        
        return all_results_dict | {"file_name" : self.file_name, "date" : timestamp}


    def test_missing_intervals(self, get_report=True) -> dict:
        """
        Метод оценивает наличие пропусков в кривых ГИС

        Required data:
            self.las_df (Pandas.DataFrame): датафрейм содержащий кривые ГИС скважины

        Args:
            get_report (bool, optional): Определяет, нужно ли отображать отчет. Defaults to True.

        Returns:
            dict: Словарь с ключевыми данными о прохождении теста
        """        

        all_results_dict = {}
        for gis_name in self.las_df.columns:
            results_dict = {}
            gis_data = self.las_df[gis_name].to_numpy()
            missing_intervals_ind = find_missing_intervals(gis_data)
            if missing_intervals_ind:
                result = False
                missing_intervals_dept = [(self.las_df.index[m_i_i[0]], self.las_df.index[m_i_i[1]]) for m_i_i in missing_intervals_ind]
                text = f'"{gis_name}": выявлены пропуски на следующих интервалах глубин:{missing_intervals_dept}'
                report_text = self.generate_report_text(text, 0)
    
            else:
                result = True
                text = f'"{gis_name}": пропуски в данных отсутствуют'
                report_text = self.generate_report_text(text, 1)
            
            results_dict['result'] = result
            results_dict['report_text'] = report_text
            all_results_dict[gis_name] = results_dict

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_text += f"{timestamp:10} / test_missing_intervals:\n{report_text}\n\n"
            if get_report: print('\n'+report_text+self.delimeter)
            
        return all_results_dict | {"file_name" : self.file_name, "date" : timestamp}


    def test_repeat(self, get_report=True) -> dict:
        """
        Метод оценивает перекрытие интервалов записи для основной и повторной записи ГИС
        Внимание! Тест не работает с неопознанными ГИС (self.unidentified_gis)

        Required data:
            self.las_df (Pandas.DataFrame): датафрейм содержащий кривые ГИС скважины

        Args:
            get_report (bool, optional): Определяет, нужно ли отображать отчет. Defaults to True.

        Returns:
            dict: Словарь с ключевыми данными о прохождении теста
        """        
        all_results_dict, count_names_dict = {}, {}
        unic_gis_names = [self.find_mnemonic(gis_name) for gis_name in self.las_df.columns if gis_name not in self.unidentified_gis]
        gis_names = [gis_name for gis_name in self.las_df.columns if gis_name not in self.unidentified_gis]

        for unic_gis_name, gis_name in zip(unic_gis_names, gis_names):
            count_names_dict[unic_gis_name] = count_names_dict.get(unic_gis_name, []) + [gis_name]   

        duplicates = [key for key, value in count_names_dict.items() if len(value) > 1]
        if duplicates:
            all_results_dict['result'] = False

            for key in duplicates: 
                logs_to_check = count_names_dict[key]
                intervals = find_depths_with_multiple_logs(self.las_df, logs_to_check)
                all_results_dict[key] = intervals
                if get_report: plot_all_logs_with_overlap(self.las_df, intervals, logs_to_check)

            all_results_dict['duplicates'] = duplicates
            text = f"Найдены дубликаты следующих ГИС: {', '.join(duplicates)}"
            report_text = self.generate_report_text(text, 0)
        else:
            text = 'Дубликатов ГИС в данных не найдено'
            report_text = self.generate_report_text(text, 1)
            if get_report: print('\n'+report_text+self.delimeter)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.report_text += f"{timestamp:10} / test_repeat:\n{report_text}\n\n"
        
        return all_results_dict | {"file_name" : self.file_name, "date" : timestamp}


    def test_surface_bounds_incoming(self, get_report=True) -> dict:
        """
        Метод оценивает вхождение интервалов отбивок пластов в интервалы снятия ГИС
        Внимание! Тест не работает с неопознанными ГИС (self.unidentified_gis)

        Required data:
            self.las_df (Pandas.DataFrame): датафрейм содержащий кривые ГИС скважины
            self.well_tops_df (Pandas.DataFrame): датафрейм содержащий отбивки пластов скважины

        Args:
            get_report (bool, optional): Определяет, нужно ли отображать отчет. Defaults to True.

        Returns:
            dict: Словарь с ключевыми данными о прохождении теста
        """        
        all_results_dict = {}
        if hasattr(self, 'well_tops_df'):
            gis_min, gis_max = self.las_df.index.min(), self.las_df.index.max()
            well_min, well_max = self.well_tops_df.MD.min(), self.well_tops_df.MD.max()
            
            result = gis_min < well_min and gis_max > well_max
            all_results_dict['result'] = str(result)

            if result:
                text = 'Отметки пластопересечений входят в интервал проведения ГИС'
                report_text = self. generate_report_text(text, 1)

            else:
                text = f"Отметки пластопересечений не входят в интервал проведения ГИС"
                report_text = self. generate_report_text(text, 0)
                report_text = f"{self.ident}Тест не пройден. \n{self.ident}{text} "

        else:
            all_results_dict['result'] = 'Fail'
            text = f"Файл с отметки пластопересечений отсутствует"
            report_text = self. generate_report_text(text, 2)

        if get_report: print('\n'+report_text+self.delimeter)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.report_text += f"{timestamp:10} / test_surface_bounds_incoming:\n{report_text}\n\n"
        
        return all_results_dict | {"file_name" : self.file_name, "date" : timestamp}