import numpy as np
import lasio
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import re
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from IPython.display import Image, display




class QA_QC_GIS_second:
    def __init__(self, las_path: str, bounds: tuple, poro_open: np.ndarray=None, perm: np.ndarray=None, poro_eff: np.ndarray=None, lithology: np.ndarray=None, depth: np.ndarray=None) -> None:
        """_summary_

        Args:
            las_path (str): Путь к .las файлу с каротажами\n
            bounds (tuple): Отбивки рассматриваемого пласта в формате (top, bottom).\n
            poro_open (np.array, optional): Открытая пористость по керну. Defaults to None.\n
            perm (np.array, optional): Проницаемость по керну. Defaults to None.\n
            poro_eff (np.array, optional): Эффективная пористость по керну. Defaults to None.\n
            lithology (np.array, optional): Литологическое описание по керну. Defaults to None.\n
            depth (np.array, optional): Глубина отбора керна.\n
        """
        self.file = open('test_result.txt', 'w')           
        self.las = lasio.read(las_path)
        self.las_depth_unit = self.las.curves[0].unit
        self.las = self.las.df()
        self.mnemonics = {'sp': ['SP', 'PS', 'ПС', 'СП', 'PS_1', 'PS_2'],
                    'gr': ['GR', 'GK', 'ГК', 'ECGR', 'GK_1', 'GK_2'],
                    'ds': ['DS', 'CALI', 'HCAL', 'CALIP', 'DCAV', 'DSN', 'DS_1', 'DS2'],
                    'mds': ['MCAL', 'MDS'],
                    'minv': ['МГЗ', 'MGZ'],
                    'mnor': ['МПЗ', 'MPZ'],
                    'mll': ['MLL', 'MBK', 'МБК', 'МКЗ', 'MSFL', 'RXOZ'],
                    'rhob': ['RHOB', 'PL', 'GGKP', 'RHOZ', 'DRHB', 'ROBB'],
                    'pef': ['PEF', 'PE', 'PEFZ', 'ZEFF'],
                    'dt': ['AK', 'DT', 'DTp', 'DTP', 'АК', 'AK_2', 'DTL', 'DTS', 'DTP1'],
                    'ild': ['IK', 'ILD', 'ИК', 'CILD', 'IKA', 'ILDA', 'IK_1', 'R27PC_46PH'],
                    'ildr': ['IKR', 'ILDR'],
                    'vikiz': ['ВИКИЗ', 'VIKIZ', 'F05', 'F07', 'F10', 'F14', 'F20', 'R05', 'R07', 'R10', 'R14', 'R20', 'С05', 'С07', 'С10', 'С14', 'С20', 'AT10', 'AT20', 'AT30', 'AT60', 'AT90', 'AF10', 'AF20', 'AF30', 'AF60',
                        'AF90', 'ILDVG1', 'ILDVG2', 'ILDVG3', 'ILDVG4', 'ILDVG5', 'ILDVR1', 'ILDVR2', 'ILDVR3', 'ILDVR4', 'ILDVR5', 'R1', 'R2', 'R3', 'R4', 'R5', 'VIK1', 'VIK2', 'VIK3', 'VIK4', 'VIK5', 'RO05', 'RO07',
                        'RO10', 'RO14', 'RO20', 'IK1', 'IK2', 'IK3', 'IK4', 'IK5', 'С07', 'С10', 'С14', 'С20', 'AT10', 'AT20', 'AT30', 'AT60', 'AT90', 'AF10', 'AF20', 'AF30', 'AF60', 'AF90', 'ILDVG1', 'ILDVG2',
                        'ILDVG3', 'ILDVG4', 'ILDVG5', 'ILDVR1', 'ILDVR2', 'ILDVR3', 'ILDVR4', 'ILDVR5', 'R1', 'R2', 'R3', 'R4', 'R5', 'VIK1', 'VIK2', 'VIK3', 'VIK4', 'VIK5', 'RO05', 'RO07', 'RO10', 'RO14', 'RO20',
                        'IK1', 'IK2', 'IK3', 'IK4', 'IK5'],
                    'ngr': ['NEUT', 'NGR', 'NGK', 'NGK_1'],
                    'nktd': ['NKTD', 'CFTC', 'NKT', 'NKTB', 'NKT_1', 'NKTB2'],
                    'nkts': ['NKTS', 'CNTC', 'NKTM', 'NKT_2'],
                    'w': ['NPHI','W','TNPH','NPLS','NPSS','TNPD','TNPL','TNPS','TNPH','TNPH_DOL','TNPH_LIM','TNPH_SAN'],
                    'depth': ['DEPT', 'MD'],
                    'gz': ['GZ']}  
        self.gis, self.missing = self.gis_preparing(top = bounds[0], bottom = bounds[1])
        self.depth = depth
        self.poro_open = poro_open
        self.perm = perm
        self.poro_eff = poro_eff
        self.lithology = lithology
        self.bounds = bounds 
        self.mnemonics_names = self.mnemonics.keys() 
        self.test_number = 0
        print('Данные каротажи не были распознаны ', self.missing)

    def check_input(self, array, param_name: str, test_name: str) -> bool:
        """Функция для проверки входных данных

            Args:
                self.data (array[T]): входной массив для проверки данных
                param_name (str): Название параметра
                test_name (str): Название теста

            Returns:
                bool: результат выполнения теста
        """

        if not isinstance(array, np.ndarray):
            self.test_number = self.test_number + 1
            self.file.write(f"{self.test_number}. Тест {test_name} не запускался. Причина {param_name} не является массивом\n")
            return False
        if len(array) == 0:
            self.test_number = self.test_number + 1
            self.file.write(f"{self.test_number}. Тест {test_name} не запускался. Причина {param_name} пустой\n")
            return False
        if True in pd.isnull(array): 
            self.test_number = self.test_number + 1
            self.file.write(f"{self.test_number}. Тест {test_name} некорректен. Причина: {param_name} состоит из NaN\n")
            return False
        if type(array[1]) == (int or float):
            for element in array:
                if not isinstance(element, (int, float)):
                    self.test_number = self.test_number + 1
                    self.file.write(
                        f"{self.test_number}. Тест {test_name} не запускался. Причина {param_name} содержит данные типа не int/float\n")
                    return False
        elif type(array[1]) == str:
            for element in array:
                if not isinstance(element, (str)):
                    self.test_number = self.test_number + 1
                    self.file.write(
                        f"{self.test_number}. Тест {test_name} не запускался. Причина {param_name} содержит данные типа не str\n")
                    return False
        return True


    def gis_preparing(self, top: float, bottom: float) -> dict:
        """Функция, используя мнемоники, определяет, какие каротажи есть в .las файле. Обрезает каротажи по отбивкам кровли и подошвы пласта

        Args:
            top (float): Глубина верхней границы интересующего интервала\n
            bottom (float): Глубина нижней границы интересующего интервала\n
        Returns:
            dict: Словарь в формате key: Мнемоника каротажа, value: Каротаж.
        """        
        data = self.las[(self.las.index > top) & (self.las.index < bottom)]
        units_list = [i.upper() for i in data]
        gis = {}
        repeat = {}
        missing = []
        missingnames = []


        for mnemonic in self.mnemonics.keys():
            unit0 = []
            for unit in units_list:
                unit1 = re.sub(r'(_\d+)?(\d+)?', '', unit)
                if unit1 in self.mnemonics.get(mnemonic):
                    unit0.append(unit)
                    gis[mnemonic] = np.array(data[unit])
                    missing.append(unit)
                if unit0 and len(unit0)>1:
                    repeat[mnemonic] = unit0
                    gis[mnemonic] = np.array(data[unit])
        missingnames = units_list

        for i in missing:
            missingnames.remove(i)
        
        gis['depth'] = data.index.to_numpy()

        self.repeat = repeat

        if gis['depth'][1] - gis['depth'][0] != 0.01:
            start = round(gis['depth'][0], 2)
            stop = round(gis['depth'][-1], 2)
            step = round(gis['depth'][1] - gis['depth'][0],2)
            interp_linspace = np.linspace(start, stop, round((step)/0.01*len(gis['depth']))-round((step/0.01-1)))
            for i in gis.keys():
                if i != 'depth':
                    gis[i] = np.interp(interp_linspace, gis['depth'], gis[i])
        gis['depth'] = np.array([round(x,2) for x in interp_linspace])


        return gis, missingnames
    
    def mnemonics_add(self, mnemonic_name, mnemonic_unit):
        """Функция добавляет в словарь мнемоник дополнительные названия

        Args:
            mnemonic_name (_type_): Название мнемоники в нижнем регистре. \n
            mnemonic_unit (_type_): Мненика, которую необходимо добавить в верхнем регистре.\n
        """        
        self.mnemonics[mnemonic_name].append(mnemonic_unit)
        self.gis, self.missing = self.gis_preparing(top = self.bounds[0], bottom = self.bounds[1])

    
    
    def kernpreproc(self, kern_arg: int=None, kern_silt: int=None, kern_sand: int=None, kern_coal: int=None) -> np.ndarray:
        """Функция преобразует литологию по керну

        Args:
            kern_arg (int, optional): Обозначение литотипа аргиллит. Defaults to None.\n
            kern_silt (int, optional): Обозначения литотипа алевролит. Defaults to None.\n
            kern_sand (int, optional): Обозначение литотипа песчаник. Defaults to None.\n
            kern_coal (int, optional): Обозначение литотипа уголь. Defaults to None.\n

        Returns:
            np.ndarray: массив с преобразованными обозначениями литологии
        """        
        kern_lithology = []
        labels = []
        if not kern_arg and not kern_silt and not kern_sand and not kern_coal:

            for i in self.lithology:
                if i[0] == 'П':
                    kern_lithology.append(1)
                    labels.append('Песчаник')
                elif i[0:2] == 'Ал':
                    kern_lithology.append(2)
                    labels.append('Алевролит')
                elif i[0:2] == 'Ар' or i[0:2] == 'Гл': 
                    kern_lithology.append(3)
                    labels.append('Аргиллит')
                elif i[0:2] == 'Уг':
                    kern_lithology.append(4)
                    labels.append('Уголь')
                else:
                    kern_lithology.append(0)
                    labels.append('Не прочитан')
        
        else:
            for i in self.lithology:
                if i == kern_sand:
                    kern_lithology.append(1)
                    labels.append('Песчаник')
                elif i == kern_silt:
                    kern_lithology.append(2)
                    labels.append('Алевролит')
                elif i == kern_arg: 
                    kern_lithology.append(3)
                    labels.append('Аргиллит')
                elif i == kern_coal:
                    kern_lithology.append(4)
                    labels.append('Уголь')
                else:
                    kern_lithology.append(0)
                    labels.append('Не прочитан')


        return np.array(kern_lithology), labels
    
    def test_lithology(self, siltmin: float = 0.4, siltmax: float = 0.6, sandmin: float = 0.1, sandmax: float = 0.4, argillitemin: float = 0.6, argillitemax: float = 1.01, coalmin = 0, coalmax=0.1, distance: float=50, minimum=None, maximum=None, gis_type: str=None, kern_arg: int=None, kern_silt: int=None, kern_sand: int=None, kern_coal: int=None, kern_shale: int=None) -> None:
        """Тест предназначен для оценки качества увязки литологии по керну и по ГИС в интервале рассматриваемого объекта. Для определения литологии по ГИС используется ПС каротаж (если ПС нет, то используется ГК), далее РИГИС литологии сопоставляется с литологией по керну на одинаковых глубинах.

        Args:
            siltmin (float, optional): Левая граница интервала амплитуд ПС и ГК, соответствующая алевролитам. Defaults to 0.4.\n
            siltmax (float, optional): Правая граница интервала амплитуд ПС и ГК, соответствующая алевролитам. Defaults to 0.7.\n
            sandmin (float, optional): Левая граница интервала амплитуд ПС и ГК, соответствующая песчанику. Defaults to 0.1.\n
            sandmax (float, optional): Правая граница интервала амплитуд ПС и ГК, соответствующая песчанику. Defaults to 0.4.\n
            argillitemin (float, optional): Левая граница интервала амплитуд ПС и ГК, соответствующая аргиллиту. Defaults to 0.7.\n
            argillitemax (float, optional): Правая граница интервала амплитуд ПС и ГК, соответствующая аргиллиту. Defaults to 1.\n
            distance (float, optional): Для нормализации значений каротажа необходимо взять значения выше и ниже рассматриваемого пласта, данный атрибут указывает, на сколько метров выше и ниже. Defaults to 50.\n
            kern_arg (int, optional): Обозначение литотипа аргиллит. Defaults to None.\n
            kern_silt (int, optional): Обозначения литотипа алевролит. Defaults to None.\n
            kern_sand (int, optional): Обозначение литотипа песчаник. Defaults to None.\n
            kern_coal (int, optional): Обозначение литотипа уголь. Defaults to None.\n
            minimum (float. optional): Минимальное значение каротажа, по которому делается РИГИС литологии, используется для нормализации.\n
            maximum (float. optional): Максимальное значение каротажа, по которому делается РИГИС литологии, используется для нормализации.\n
            gis_type (str, optional): Каротаж, по которому делается РИГИС литологии. Defaults to 'gr'.\n
            coalmin (float, optional): Левая граница интервала амплитуд ПС и ГК, соответствующая углям. Defaults to 0.\n
            coalmax (float, optional): Левая граница интервала амплитуд ПС и ГК, соответствующая углям. Defaults to 0.1.\n
            

        """      
        if ('sp' or 'gr' in self.gis.keys()) and self.check_input(self.lithology, 'Литология по керну', 'Увязка керн и ГИС по литологии') and self.check_input(self.depth, 'Глубина отбора керна', 'Увязка керн и ГИС по литологии'):
             
            kern_lithology, labels = self.kernpreproc(kern_arg, kern_silt, kern_sand, kern_coal)
            gis_lithology, missing = self.gis_preparing(top=self.bounds[0]-distance, bottom=self.bounds[1]+distance)

            if 'sp' in self.gis.keys():
                minimum1 = min(gis_lithology['sp'])
                maximum1 = max(gis_lithology['sp'])
                asp = np.array([(x-minimum1)/(maximum1-minimum1) for x in gis_lithology['sp']])
            if 'gr' in self.gis.keys():
                minimum1 = min(gis_lithology['gr'])
                maximum1 = max(gis_lithology['gr'])
                dgr = np.array([(x-minimum1)/(maximum1-minimum1) for x in gis_lithology['gr']])


            if 'sp' in self.gis.keys() and gis_type=='sp':
                if not minimum:
                    minimum = min(gis_lithology['sp'])
                if not maximum:
                    maximum = max(gis_lithology['sp'])
                log_for_lithology = np.array([(x-minimum)/(maximum-minimum) for x in gis_lithology['sp']])
            elif 'gr' in self.gis.keys(): 
                if not minimum:
                    minimum = min(gis_lithology['gr'])
                if not maximum:
                    maximum = max(gis_lithology['gr'])
                log_for_lithology = np.array([(x-minimum)/(maximum-minimum) for x in gis_lithology['gr']])
            
            depth_gis = gis_lithology['depth']
            lithology_rigis = []
            count_matches = 0
            count = 0
            dgrforplot = []
            transparency = []
            depth = []
            kernlitho = []
            aspforplot = []
            kern_labels = []
            rigis_labels= []

            lithology_ranges = {
                1: (sandmin, sandmax, 'Песчаник'),
                2: (siltmin, siltmax, 'Алевролит'),
                3: (argillitemin, argillitemax, 'Аргиллит'),
                4: (coalmin, coalmax, 'Уголь')
            }
            for i, kern_depth in enumerate(self.depth):
                if kern_depth > 0 and round(kern_depth, 1) in depth_gis:
                    index = np.where(depth_gis == round(kern_depth, 2))[0][0]
                    if log_for_lithology[index] >= 0 and kern_depth < depth_gis[-1]:
                        count += 1

                        if 'sp' in self.gis.keys():
                            aspforplot.append(asp[index])
                        if 'gr' in self.gis.keys():
                            dgrforplot.append(dgr[index])

                        depth.append(depth_gis[index])
                        log_value = log_for_lithology[index]
                        result = []
                        result1 = []
                        for lithology_value, (lower_bound, upper_bound, lithology_label) in lithology_ranges.items():
                            if lower_bound <= log_value < upper_bound and kern_lithology[i] == lithology_value:
                                count_matches = count_matches + 1
                                transparency.append(0.001)
                                result.append(True)
                            else:
                                result.append(False)
                            if lower_bound <= log_value < upper_bound:
                                lithology_rigis.append(lithology_value)
                                kernlitho.append(kern_lithology[i])
                                rigis_labels.append(labels[i])
                                kern_labels.append(lithology_label)
                                result1.append(True)
                            else:
                                result1.append(False)


                        if True not in result:
                            transparency.append(1)
                        if True not in result1:
                            lithology_rigis.append(0)
                            kernlitho.append(kern_lithology[i])
                            kern_labels.append('Прочее')
                            rigis_labels.append(labels[i])

            
            self.lithology_test_visualization(dgrforplot, transparency, kernlitho, lithology_rigis, count_matches, count, depth, aspforplot, kern_labels, rigis_labels)
    
    
    def lithology_test_visualization(self, dgrforplot: list, prozr: list, kern_lithology: list, litho: list, count: int, count1: int, depth: list, aspforplot: list, kern, rigis):
            
            """Функция визуализирует результаты теста увязки литологии по керну и по ГИС. Отображает каротаж ГИС и отмечает на нем точки, в которых литология не увязана. Также отображает литологию по керну. Атрибуты функции указывать не нужно

            Args:
                dgrforplot (list): dGR каротаж.\n
                prozr (list): Для визуализации точек, в которых литология по керну и ГИС не сходится.\n
                kern_lithology (list): Литология по керну.\n
                litho (list): РИГИС литологии.\n
                count (int): Количество увязанных точек.\n
                count1 (int): Общее количество точек.\n
                depth (list): Глубины, на которых проверялась увязка.\n
                aspforplot (list): Каротаж aSP.\n
            """         
            
            print('Процент совпавших литотипов по ГИС и по керну равен ', str(round(count/count1*100,3)), '%')
            self.test_number = self.test_number + 1 
            self.file.write(str(self.test_number) + '. Тестирование качества увязки литологии по ГИС и литологии по керну.\nПроцент совпавших литотипов по ГИС и по керну равен ' + str(count/count1*100) + ' %.\n')
            print('Желтыми точками отмечены глубины, в которых литология не увязана.')

            column_width = [0.75, 0.125, 0.125]
            column_titles = ['dGR', 'Керн', 'ГИС']
            colors = [
                "rgb(255, 250, 205)",
                "rgb(139, 69, 19)",
                "rgb(0, 0, 0)",
                "rgb(255, 250, 205)",
                "rgb(240, 230, 140)",
                ] 
            colorscale=[
                    (0.0, colors[3]),
                    (0.25, colors[0]),
                    (0.5, colors[4]),
                    (0.75, colors[1]),
                    (1, colors[2]),
                ]
            kern_lithology = [(x-0)/(4-0) for x in kern_lithology]
            litho = [(x-0)/(4-0) for x in litho]
            if 'sp' in self.gis.keys():
                m=4
                column_width = [0.3, 0.3, 0.2, 0.2]
                column_titles = ['dGR','aSP', 'Керн', 'ГИС']
            elif 'gr' in self.gis.keys():
                m=3
                column_width = [0.75, 0.125, 0.125]
                column_titles = ['dGR', 'Керн', 'ГИС']
            fig = make_subplots(rows=1, cols=m, column_widths=column_width, subplot_titles=column_titles, horizontal_spacing=0.1)
            fig.add_trace(go.Scatter(x=dgrforplot, y=depth, mode='lines+markers',line=dict(color='grey'), marker=dict(opacity=prozr, color='orange'), hovertemplate='Глубина: %{y} <br> Значение каротажа: %{x:.2f}'), row=1, col=1)
            fig.add_trace(go.Heatmap(y=depth, z=np.array(kern_lithology).reshape(-1, 1), hovertext=np.array(rigis).reshape(-1, 1), colorscale=colorscale, showscale=False, zmin=0, zmax=1, hovertemplate='Глубина: %{y} <br>Литология: %{hovertext}'), row=1, col=m-1)
            fig.add_trace(go.Heatmap(y=depth, z=np.array(litho).reshape(-1, 1), hovertext=np.array(kern).reshape(-1, 1), colorscale=colorscale, showscale=False, zmin=0, zmax=1, hovertemplate='Глубина: %{y} <br>Литология: %{hovertext}'), row=1, col=m)
            if 'sp' in self.gis.keys():
                fig.add_trace(go.Scatter(x=aspforplot, y=depth, mode='lines+markers',line=dict(color='grey'), marker=dict(opacity=prozr, color='orange'), hovertemplate='Глубина: %{y} <br> Значение каротажа: %{x}'), row=1, col=2)
                n=5
            else:
                n=4
            for i in range(1, n):
                fig.update_yaxes(autorange="reversed", row=1, col=i, matches='y')
            for i in range(n-2,n):
                fig.update_xaxes(showticklabels=False, row=1, col=i)
            for i in fig.data:
                i.hoverlabel = {'namelength' : 0}

            fig.update_layout(height=1000, width=600, yaxis_range=[depth[0], depth[-1]], xaxis_range=[0,1]) 
            fig.show()
            display(Image(filename='data/legend1.png', height=300, width=600))

    def properties(self, poro_model = None, poroeff_model = None, perm_model = None, gis_type1 = 'rhob', gis_type2 = None, poro_perm_type='poroeff') -> pd.core.frame.DataFrame:
        """Функция создает РИГИС пористости, эффективной пористости и проницаемости.

        Args:
            poro_model (_type_, optional): Модель пористости. Defaults to None.\n
            poroeff_model (_type_, optional): Модель эффективной пористости. Defaults to None.\n
            perm_model (_type_, optional): Модель проницаемости. Defaults to None.\n
            gis_type1 (str, optional): Название каротажа ГИС, по которому считается пористость. Defaults to 'rhob'.\n
            gis_type2 (_type_, optional): Название дополнительного каротажа ГИС, по которому считается пористость . Defaults to None.\n
            poro_perm_type (str, optional): Название свойства, по которому считается проницаемость. Defaults to 'poroeff'.\n

        Returns:
            pd.core.frame.DataFrame: Таблица с РИГИС пористости, эффективной пористости и проницаемости
        """
        
        poro_model = poro_model or self.poro_model
        poroeff_model = poroeff_model or self.poroeff_model
        perm_model = perm_model or self.perm_model
        poro_perm_type = poro_perm_type or 'poroeff'
        gis_type2 = gis_type2 or None 

        rigis = pd.DataFrame(columns=['poro', 'poroeff', 'perm', 'depth'])
        depth = self.gis['depth']
        gis_for_properties = self.gis[gis_type1]
        kern_poro = []
        rigis_poro = []
            
        for i in range(0, len(self.depth)):
            if self.depth[i] > 0 and round(self.depth[i], 2) in depth and i<len(self.poro_open):
                    depthindex = np.where(depth == round(self.depth[i], 2))[0][0]
                    if gis_for_properties[depthindex] > 0 and self.poro_open[i] > 0:
                        o = len(rigis['poro'])
                        rigis.at[o, 'poro'] = (poro_model(self.gis[gis_type1][depthindex]) if not gis_type2 else poro_model( self.gis[gis_type2][depthindex], gis_for_properties2[depthindex]))
                        rigis.at[o, 'depth'] = depth[depthindex]
                        kern_poro.append(self.poro_open[i])
                        rigis_poro.append(rigis.at[o, 'poro'])
                        rigis.at[o, 'poroeff'] = poroeff_model(rigis.loc[o]['poro'])
                        rigis.at[o, 'perm'] = perm_model(rigis.loc[o][poro_perm_type])
    

        if any(x > 1 for x in rigis_poro):
            rigis_poro = [x / 100 for x in rigis_poro]
        if any(x > 1 for x in kern_poro):
            kern_poro = [x / 100 for x in kern_poro]


        return rigis, np.array(kern_poro), np.array(rigis_poro)
        
    
    def poro_model(self, rhob, poro_koeff1 = -49.72, poro_koeff2 = 135.69) -> float:
        """
            Функция представляет собой петрофизическую модель пористости вида poro = poro_koeff1*rhob + poro_koeff2
        Args:
            poro_koeff1 (float): Множитель перед RHOB,\n
            poro_koeff2 (float): Свободный член уравнения.\n
        Returns:
            float: значение РИГИС пористости
        """
        return poro_koeff1*rhob + poro_koeff2

    def poroeff_model(self, poro, poroeff_koeff1 = 1/0.5674, poroeff_koeff2 = -12.7827/0.5674) -> float:
        """
            Функция представляет собой петрофизическую модель эффективной пористости вида poroeff = poro_koeff1*poro + poro_koeff2
        Args:
            poroeff_koeff1 (float): Множитель перед poro,\n
            poroeff_koeff2 (float): Свободный член уравнения.\n
        Returns:
            float: значение РИГИС эффективной пористости
        """
        return poroeff_koeff1*poro + poroeff_koeff2

    def perm_model(self, poroeff, perm_koeff1 = 0.065, perm_koeff2 = 0.44) -> float:
        """
            Функция представляет собой петрофизическую модель проницаемости вида perm = perm_koeff1*poroeff + perm_koeff2
        Args:
            perm_koeff1 (float): Множитель перед poroeff,\n
            perm_koeff2 (float): Свободный член уравнения.\n
        Returns:
            float: значение РИГИС проницаемости
        """
        return perm_koeff1*np.exp(perm_koeff2*poroeff)

    def trendline(self, prop1, prop2):
        """
            Функция строит аппроксимацию для кроссплота с выбранными данными по осям X, Y
        Args:
            prop1 (list): Данные по оси X,\n
            prop2 (list): Данные по оси Y.\n
        """
        k, b, r, p, se = stats.linregress(prop1, prop2)
        return k, b, r

    def test_porosity(self, poro_model = None, poroeff_model = None, perm_model = None, gis_type1 = 'rhob', gis_type2 = None, poro_perm_type='poroeff'):
        """Тест предназначен для оценки качества увязки результатов керновых исследований и ГИС по пористости.

        Args:
            poro_model (_type_, optional): Модель пористости. Defaults to None.\n
            poroeff_model (_type_, optional): Модель эффективной пористости. Defaults to None.\n
            perm_model (_type_, optional): Модель проницаемости. Defaults to None.\n
            gis_type1 (str, optional): Название каротажа ГИС, по которому считается пористость. Defaults to 'rhob'.\n
            gis_type2 (_type_, optional): Название дополнительного каротажа ГИС, по которому считается пористость . Defaults to None.\n
            poro_perm_type (str, optional):Название свойства, по которому считается проницаемост. Defaults to 'poroeff'.\n
        """        
        if self.check_input(self.poro_open, 'Открытая пористость по керну', 'Тестирование увязки керна и ГИС по открытой пористости'):
            rigis, self.kern_poro, self.rigis_poro = self.properties(poro_model = poro_model, poroeff_model = poroeff_model, perm_model = perm_model, gis_type1 = gis_type1, gis_type2 = gis_type2, poro_perm_type=poro_perm_type)
            self.porosity_visualization(self.kern_poro, self.rigis_poro)



    def test_permeability(self, poro_model = None, poroeff_model = None, perm_model = None, gis_type1 = 'rhob', gis_type2 = None, poro_perm_type='poroeff'):
        """Тест предназначен для оценки качества увязки результатов керновых исследований и ГИС по проницаемости.
        Args:
            poro_model (_type_, optional): Модель пористости. Defaults to None.\n
            poroeff_model (_type_, optional): Модель эффективной пористости. Defaults to None.\n
            perm_model (_type_, optional): Модель проницаемости. Defaults to None.\n
            gis_type1 (str, optional): Название каротажа ГИС, по которому считается пористость. Defaults to 'rhob'.\n
            gis_type2 (_type_, optional): Название дополнительного каротажа ГИС, по которому считается пористость . Defaults to None.\n
            poro_perm_type (str, optional):Название свойства, по которому считается проницаемост. Defaults to 'poroeff'.\n
        """        
        
        if self.check_input(self.perm, 'Проницаемость по керну', 'Тестирование увязки керна и ГИС по проницаемости'):
            rigis, kern_poro, rigis_poro = self.properties(poro_model = poro_model, poroeff_model = poroeff_model, perm_model = perm_model, gis_type1 = gis_type1, gis_type2 = gis_type2, poro_perm_type=poro_perm_type)
            if len(rigis.values) > 0:
                    kern_perm = []
                    rigis_perm = []

                    for i in range(len(self.perm)):
                        if self.perm[i] > 0 and round(self.depth[i], 2) in list(rigis['depth']):
                            rigis_perm.append(float(rigis.loc[np.where(rigis['depth'] == round(self.depth[i], 2))[0][0]]['perm']))
                            kern_perm.append(self.perm[i])
            self.permeability_vizualization(np.array(kern_perm), np.array(rigis_perm))
    
    def permeability_vizualization(self, kern_perm, rigis_perm):
        """Визуализация результатов тестирования проницаемости
        Args:
            kern_perm (_type_): _description_\n
            rigis_perm (_type_): _description_\n
        """        
        self.test_number = self.test_number + 1
        if list(kern_perm):
            def linear_func(x, a, b):
                return a * x + b
            k_perm, b_perm, r_perm = self.trendline(rigis_perm, kern_perm)
            y_new = linear_func(rigis_perm, k_perm, b_perm)
            plt.scatter(rigis_perm, kern_perm)
            plt.plot(sorted(rigis_perm), sorted(y_new), c='orange')
            maximum = max(max(rigis_perm), max(kern_perm))
            plt.plot([0, maximum], [0, maximum], '--', c='g')
            plt.xlabel('РИГИС проницаемости, мД')
            plt.ylabel('Проницаемость по керну, мД')
            plt.xlim(0.01, maximum*2)
            plt.ylim(0.01, maximum*2)
            plt.xscale('log')
            plt.yscale('log')
            plt.text(0.05, 0.05, 'y=' + str(round(k_perm, 3)) + '*poroeff_kern+' + str(round(b_perm,3))  + ', коэффициент R^2 = ' + str(round(r_perm, 3)))
            plt.title('Проницаемость')
            plt.show()
            self.file.write(str(self.test_number) + '. Проницаемость: Коэффициент k = ' + str(round(k_perm, 3)) + ', коэффициент R^2 = ' + str(round(r_perm, 3)) + '\n')
            
        else:
            print('В файле с керном нет таких глубин, на которых были бы данные по открытой пористости и по проницаемости')
            self.file.write('   В файле с керном нет таких глубин, на которых были бы данные по открытой пористости и по проницаемости.\n')
        
    def test_porosityeff(self, poro_model = None, poroeff_model = None, perm_model = None, gis_type1 = 'rhob', gis_type2 = None, poro_perm_type='poroeff'):
        """Тест предназначен для оценки качества увязки результатов керновых исследований и ГИС по эффективной пористости.

        Args:
            poro_model (_type_, optional): Модель пористости. Defaults to None.\n
            poroeff_model (_type_, optional): Модель эффективной пористости. Defaults to None.\n
            perm_model (_type_, optional): Модель проницаемости. Defaults to None.\n
            gis_type1 (str, optional): Название каротажа ГИС, по которому считается пористость. Defaults to 'rhob'.\n
            gis_type2 (_type_, optional): Название дополнительного каротажа ГИС, по которому считается пористость . Defaults to None.\n
            poro_perm_type (str, optional):Название свойства, по которому считается проницаемост. Defaults to 'poroeff'.\n
        """        
        if self.check_input(self.poro_eff, 'Эффективная пористость по керну', 'Тестирование увязки керна и ГИС по эффективной пористости'):
            rigis, kern_poro, rigis_poro = self.properties(poro_model = poro_model, poroeff_model = poroeff_model, perm_model = perm_model, gis_type1 = gis_type1, gis_type2 = gis_type2, poro_perm_type=poro_perm_type)
            if len(rigis.values) > 0:
                    kern_poroeff = []
                    rigis_poroeff = []
                    
                    for i in range(len(self.poro_eff)):
                        if self.poro_eff[i] > 0 and round(self.depth[i], 2) in list(rigis['depth']):
                            rigis_poroeff.append(float(rigis.loc[np.where(rigis['depth'] == round(self.depth[i], 2))[0][0]]['poroeff']))
                            kern_poroeff.append(self.poro_eff[i])
            self.porosityeff_vizualization(kern_poroeff, rigis_poroeff)

    def porosityeff_vizualization(self, kern_poroeff, rigis_poroeff):
        """Визуализация результатов тестирования увязки эффективной проницаемости

        Args:
            kern_poroeff (_type_): _description_\n
            rigis_poroeff (_type_): _description_\n
        """    
        self.test_number = self.test_number + 1    
        if list(kern_poroeff):
            k_poroeff, b_poroeff, r_poroeff = self.trendline(rigis_poroeff, kern_poroeff)
            plt.scatter(rigis_poroeff, kern_poroeff)
            plt.plot([0, sorted(rigis_poroeff)[-1]], [b_poroeff, k_poroeff*sorted(rigis_poroeff)[-1]+b_poroeff], c='orange')
            maximum = max(max(rigis_poroeff), max(kern_poroeff))
            plt.xlim(0, maximum*1.1)
            plt.ylim(0, maximum*1.1)
            plt.plot([0, maximum], [0, maximum], '--', c='g')
            plt.xlabel('РИГИС эффективной пористости')
            plt.ylabel('Эффективная пористость по керну')
            plt.text(maximum*1.1/6, maximum*1.1/6, 'y=' + str(round(k_poroeff, 3)) + '*poro_kern+' + str(round(b_poroeff,3))+ '. Коэффициент R^2 = ' + str(round(r_poroeff, 3)))
            plt.title("Эффективная пористость")
            plt.show()
            self.file.write(str(self.test_number) + '. Эффективная пористость: Коэффициент k = ' + str(round(k_poroeff, 3)) + ', коэффициент R^2 = ' + str(round(r_poroeff, 3)) + '\n')

        else:
            print('')
            print('В файле с керном нет таких глубин, на которых были бы данные по эффективной пористости и по открытой пористости')
            self.file.write(str(self.test_number) + 'В файле с керном нет таких глубин, на которых были бы данные по эффективной пористости и по открытой пористости\n')
            print('')

    def porosity_visualization(self, kern_poro, rigis_poro):
        """Визуализация результатов тестирования увязки пористости

        Args:
            kern_poro (_type_): _description_\n
            rigis_poro (_type_): _description_\n
        """        
        self.test_number = self.test_number + 1
        if list(kern_poro):
            k_poro, b_poro, r_poro = self.trendline(rigis_poro, kern_poro)
            plt.scatter(rigis_poro, kern_poro)
            plt.plot([0, sorted(rigis_poro)[-1]], [b_poro, k_poro*list(sorted(rigis_poro))[-1]+b_poro], c='orange')
            maximum = max(max(rigis_poro), max(kern_poro))
            plt.xlim(0, maximum*1.1)
            plt.ylim(0, maximum*1.1)
            plt.plot([0, maximum], [0, maximum], '--', c='g')
            plt.xlabel('Пористость по керну')
            plt.ylabel('РИГИС пористости')
            plt.text(maximum*1.1/6, maximum*1.1/6, 'y=' + str(round(k_poro, 3)) + '*poro_kern+' + str(round(b_poro,3)) + '. Коэффициент R^2 = ' + str(round(r_poro, 3)))
            plt.title('Пористость')
            plt.show()
            self.file.write(str(self.test_number) + '. Пористость: Коэффициент k = ' + str(round(k_poro, 3)) + ', коэффициент R^2 = ' + str(round(r_poro, 3)) + '\n')

        else:
            print('В файле с керном нет данных по пористости')
            self.file.write(str(self.test_number) + '. В файле с керном нет данных по пористости\n')

        
    def test_skipped_gis(self, delta: float=0.5) -> None:
        """Тест направлен на поиск пропусков в записи ГИС, и, в случае интервала пропусков меньше delta м, их заполнения интерполяцией.

        Args:
            delta (float, optional): Атрибут задает максимальный интервал в метрах, в пределах которого пропуски будут заполняться интерполяцией . Defaults to 0.5.
        """
        self.test_number = self.test_number + 1
        self.file.write(str(self.test_number) + '. Тест на наличие пропусков в записи ГИС\n')
        for gis in self.gis.keys():
            is_missing = np.isnan(self.gis[gis])
            split_indices = np.where(np.diff(np.concatenate(([0], is_missing, [0]))) != 0)[0]
            difference = 1/(self.gis['depth'][1] - self.gis['depth'][0]) * delta

            if list(split_indices) and all(np.diff(np.array(split_indices))[::2] <= difference):
                print('В каротаже ' + gis + ' пропуски на следующих глубинах:')
                self.file.write('В каротаже ' + gis + ' пропуски на следующих глубинах:\n')

                for i in range(1,int((len(split_indices))/2+1)):
                    print(str(self.gis['depth'][split_indices[i*2-2]]) + ' - ' + str(self.gis['depth'][split_indices[i*2-1]]))
                    self.file.write(str(self.gis['depth'][split_indices[i*2-2]]) + ' - ' + str(self.gis['depth'][split_indices[i*2-1]]) + '\n')
                    new = self.interpolation_gis(split_indices[i*2-2], split_indices[i*2-1], gis)
                    self.gis[gis][split_indices[i*2-2]:split_indices[i*2-1]-1] = new
                    
            elif list(split_indices):
                print('В каротаже ' + gis + ' пропуски больше 0.5м. \n') 
                self.file.write('В каротаже ' + gis + ' пропуски больше 0.5м. \n') 
        
    
    def interpolation_gis(self, start: int, stop: int, i: str) -> np.ndarray:
        """Интерполяция пропущенного интервала ГИС

        Args:
            start (_type_): _description_\n
            stop (_type_): _description_\n
            i (_type_): _description_\n

        Returns:
            _type_: _description_
        """
        gismin = self.gis['depth'][start]
        gismax = self.gis['depth'][stop]
        diff = self.gis['depth'][1] - self.gis['depth'][0]
        new = np.interp(np.linspace(gismin, gismax - diff, int((gismax-gismin)/diff)), [self.gis['depth'][start-1], gismax], [self.gis[i][start-1], self.gis[i][stop]])
        return new
    
    def test_repeat(self) -> None:
        """Тест направлен на поиск интервалов замещения ГИС. Если такие есть, то значение каротажа в этом промежутке задается как среднее.
        """        
        self.test_number = self.test_number + 1
        if self.repeat:
            print('Перекрытие интервалов есть в каротажах: ')
            self.file.write(str(self.test_number) + '. Перекрытие интервалов есть в каротажах: \n')
            for keys, values in self.repeat.items():
                print(keys)
                self.file.write(keys)
                for count in range(len(self.las[values[0]])):
                    value = []
                    for o in self.repeat[keys]:
                        value.append(self.las.loc[self.las.index[count]][o])
                    self.gis[keys][count] = np.mean(value)
                        
        else:
            print('Перекрытия интервалов в данных нет')
            self.file.write(str(self.test_number) + '. Перекрытия интервалов в данных нет\n')
        

    def test_max_value_gis(self) -> None:
        """
        Тест предназначен для проверки соответствий отметок пластопересечений и глубинного диапазона ГИС.
        """
        self.test_number = self.test_number + 1
        depth = self.las.index
        if depth[-1] <= self.bounds[1]:
            print('Максимальная глубина по отбивке ' + str(self.bounds[1]) + ', максимальная глубина по ГИС ' + str(depth[-1]) + str('. Тест не пройден'))
            self.file.write(str(self.test_number) + '. Тестирование на сходство максимальных отметок глубин по отбивкам и по ГИС.\nМаксимальная глубина по отбивке ' + str(self.bounds[1]) + ', максимальная глубина по ГИС ' + str(depth[-1]) + str('. Тест не пройден\n'))
        else:
            print('Максимальная глубина по отбивке ' + str(self.bounds[1]) + ', максимальная глубина по ГИС ' + str(depth[-1]) + str('. Тест пройден'))
            self.file.write(str(self.test_number) + '. Тестирование на сходство максимальных отметок глубин по отбивкам и по ГИС. \nМаксимальная глубина по отбивке ' + str(self.bounds[1]) + ', максимальная глубина по ГИС ' + str(depth[-1]) + str('. Тест пройден\n'))
            

    def saturation_test(self, Archi_model= None, J_model = None, gis_type1: str = 'ild', Pc: float= None, deltadens: float=None, h: float=None, sigma: float= None, tetta: float= None, poro_model = None, poroeff_model = None, perm_model = None, gis_type: str= 'rhob', gis_type2: str=None):
        """Тест предназначен для сопоставления моделей водонасыщенности(Арчи, J функция, ОФП) и оценки качества их увязки с данными по керну.

        Args:
            Archi_model (function, optional): Модель водонасыщенности Арчи. Defaults to None.\n
            J_model (function, optional): Уравнение аппроксимации кроссплота, где x-J, y-Sw. Defaults to None.\n
            gis_type1 (str, optional): Мнемоника каротажа ГИС, который используется в модели Арчи. Defaults to 'ild'.\n
            Pc (float, optional): Капиллярное давление. Defaults to None.\n
            sigma (float, optional): sigma. Defaults to None.\n
            tetta (float, optional): Угол смачивания. Defaults to None.\n
            poro_model (function, optional): Модель пористости. Defaults to None.\n
            poroeff_model (function, optional): Модель эффективной пористости. Defaults to None.\n
            perm_model (function, optional): Модель проницаемости. Defaults to None.\n
            gis_type (str, optional): Мнемоника каротажа ГИС, по которому считается пористость. Defaults to 'rhob'.\n
        """        
        if (Pc or (deltadens and h)) and sigma and tetta:

            rigis = self.properties(poro_model = poro_model, poroeff_model = poroeff_model, perm_model = perm_model, gis_type1 = gis_type, gis_type2=gis_type2)
            
            if not Archi_model:
                Archi_model = self.Archi_model
            
            if not J_model:
                J_model = self.J_function

            if not Pc:
                Pc = deltadens*9.81*h/(sigma*np.cos(tetta))
            
            Archi = []
            J_func = []

            for i in len(rigis.values()):
                porosity = rigis.loc[i]['poro']
                permeability = rigis.loc[i]['perm']
                Archi.append(Archi_model(self.gis[gis_type1][np.where(self.gis['depth'] == rigis.loc[i]['depth'])], porosity))
                J_func.append(J_model(Pc*(permeability/porosity)**0.5/(sigma*np.cos(tetta))))

            self.saturation_report(np.array(Archi), np.array(J_func))

    def test_saturation(self, poro_model, Archi_kern, J_kern, ofp_kern, Archi_model=None, J_model=None, ofp_model=None, gis_type: str = 'ild', poroeff_model = None, perm_model = None, gis_type1: str= 'rhob', gis_type2: str=None):        
        """Тест предназначен для сопоставления моделей водонасыщенности(Арчи, J функция, ОФП) и оценки качества их увязки с данными по керну.

        Args:
            poro_model (_type_): Модель пористости.\n
            Archi_kern (_type_): Водонасыщенность по керну для модели Арчи.\n
            J_kern (_type_): Водонасыщенность по керну для модели J функции.\n
            ofp_kern (_type_): Водонасыщенность по керну для модели по офп.\n
            Archi_model (_type_, optional): Модель водонасыщенности Арчи. Defaults to None.\n
            J_model (_type_, optional): Модель водонасыщенности по J функции. Defaults to None.\n
            ofp_model (_type_, optional): Модель водонасыщенности по офп. Defaults to None.\n
            gis_type (str, optional): Каротаж ГИС, использующийся в модели Арчи. Defaults to 'ild'.\n
            poroeff_model (_type_, optional): Модель эффективной пористости. Defaults to None.\n
            perm_model (_type_, optional): Модель проницаемости. Defaults to None.\n
            gis_type1 (str, optional): Каротаж ГИС, по которому считается пористость. Defaults to 'rhob'.\n
            gis_type2 (str, optional): Дополнительный каротаж ГИС, по которому считается пористость. Defaults to None.\n
        """        
        if not Archi_model:
            Archi_model = self.Archi_model    
        if not J_model:
            J_model = self.J_function
        if not ofp_model:
            ofp_model = self.ofp_model
        
        
        
        rigis, poro, poroo = self.properties(poro_model = poro_model, poroeff_model = None, perm_model = None, gis_type1 = 'sp', gis_type2=None)
                
        kern = {'Archi' : Archi_kern,
                'J' : J_kern,
                'ofp' : ofp_kern}


        saturation = pd.DataFrame(columns=['poro', 'Archi', 'J', 'ofp', 'depth'])
        depth1 = self.gis['depth']
        gis_for_properties = self.gis[gis_type]
            
        for i in rigis.index:
            if self.depth[i] > 0:
                    depthindex = np.where(depth1 == round(self.depth[i], 2))[0][0]
                    if gis_for_properties[depthindex] > 0 and self.poro_open[i] > 0:
                        saturation.at[i, 'Archi'] = Archi_model(gis_for_properties[depthindex], rigis.loc[i]['poro'])
                        saturation.at[i, 'depth'] = depth1[depthindex]
                        saturation.at[i, 'J'] = J_model(rigis.loc[i]['poro'])
                        saturation.at[i, 'ofp'] = ofp_model(rigis.loc[i]['poro'])
                        saturation.at[i, 'poro'] = rigis.loc[i]['poro']
        
        self.tables = {}
        titles = ['Модель Арчи', 'J функция', 'Модель по ОФП']
        for n, i in enumerate(saturation.columns[1:4]):
            if (n == 0 and Archi_model != self.Archi_model) or (n == 1 and J_model != self.J_function) or (n == 2 and ofp_model != self.ofp_model):
                table = pd.DataFrame(columns=['rigis', 'kern'])
                table['rigis'] = saturation[i]
                table['kern'] = kern[i]
                table = table.dropna()
                self.tables[i] = table
                self.saturation_vizualization(table['rigis'].to_list(), table['kern'].to_list(), titles[n])
            
    def saturation_vizualization(self, saturation_rigis, saturation_kern, title):
            """_summary_
            Args:
                saturation_rigis (_type_): _description_
                saturation_kern (_type_): _description_
                title (_type_): _description_
            """
            self.test_number = self.test_number + 1            
            k_sat, b_sat, r_sat = self.trendline(saturation_rigis, saturation_kern)
            plt.scatter(saturation_rigis, saturation_kern)
            plt.plot([0.01, sorted(saturation_rigis)[-1]], [b_sat, k_sat*sorted(saturation_rigis)[-1]+b_sat], c='orange')
            if max(saturation_rigis) > max(saturation_kern):
                maximum = max(saturation_rigis)
            else:
                maximum = max(saturation_kern)
            plt.plot([0, maximum], [0, maximum], '--', c='g')
            plt.xlabel('РИГИС водонасыщенности')
            plt.ylabel('Водонасыщенность по керну')
            plt.title(str(title) + '. y=' + str(round(k_sat, 3)) + '*saturation_kern+' + str(round(b_sat,3))  + ', коэффициент R^2 = ' + str(round(r_sat, 3)))
            plt.show()
            self.file.write(str(self.test_number) + '. ' + str(title) + ': Коэффициент k = ' + str(round(k_sat, 3)) + ', коэффициент R^2 = ' + str(round(r_sat, 3)) + '\n')
      
    def saturation_report(self, Archi: np.ndarray, J_func: np.ndarray) -> None:
            """ Результат тестирования качества увязки моделей водонасыщенности.

            Args:
                Archi (_type_): _description_\n
                J_func (_type_): _description_\n
            """            

            if list(Archi) and list(J_func):
                plt.scatter(Archi, J_func)
                k, b, r = self.trendline(Archi, J_func)
                plt.plot([0, sorted(Archi)[-1]], [b, k*list(sorted(Archi))[-1]+b], c='orange')
                plt.xlabel('Водонасыщенность по Арчи')
                plt.ylabel('Водонасыщенность по J-функции')
                plt.title('Коэффициент k = ' + str(k) + ', коэффициент R^2 = ' + str(r))
                plt.show()
                
            else:
                print('В файле с керном нет данных по пористости')
            
            

    def Archi_model(self, ild: float, poro: float) -> float:
        """_summary_

        Args:
            ild (_type_): _description_\n
            poro (_type_): _description_\n

        Returns:
            _type_: _description_
        """        
        return  (1.7456*1.0387*0.24/(ild*poro**1.3197))**(1/1.5885)

    def J_function(self, J: float) -> None:
        """_summary_

        Args:
            J (_type_): _description_

        Returns:
            _type_: _description_
        """        
        return 1*J-1
    
    def ofp_model(self, poro: float) -> None:
        """_summary_

        Args:
            J (_type_): _description_

        Returns:
            _type_: _description_
        """        
        return 1*poro-1
    

    def test_units(self, depth_unit: str= 'м'):
        """ Тест предназначен для проверки размерностей глубин в файле с ГИС и в файле с отбивками.

        Args:
            depth_unit (str): Единица измерения глубины по керну.
        """
        if depth_unit:
            self.test_number = self.test_number + 1
            if (self.las_depth_unit == 'м' or self.las_depth_unit == 'm') and (depth_unit == 'm' or depth_unit == 'м'):
                self.file.write(str(self.test_number) + '. В ГИС и в файле с отбивками глубина измеряется в метрах')
                print('В ГИС и в файле с отбивками глубина измеряется в метрах')
            elif (self.las_depth_unit == 'фут' or self.las_depth_unit == 'ft') and (depth_unit == 'фут' or depth_unit == 'ft'):
                self.file.write(str(self.test_number) + '. В ГИС и в файле с отбивками глубина измеряется в футах')
                print('В ГИС и в файле с отбивками глубина измеряется в футах')
            elif self.las_depth_unit == depth_unit:
                self.file.write(str(self.test_number) + '. В ГИС и в файле с отбивками глубина измеряется в ' + depth_unit)
                print('В ГИС и в файле с отбивками глубина измеряется в ' + depth_unit)
            else:
                self.file.write(str(self.test_number) + '. В ГИС глубина измеряется в ' + self.las_depth_unit + ', в файле с отбивками глубина измеряется в ' + depth_unit)
                print('В ГИС глубина измеряется в ' + self.las_depth_unit + ', в файле с отбивками глубина измеряется в ' + depth_unit)

    def get_list_of_tests(self) -> list:
        """Функция выводит список доступных тестов.

        Returns:
            list: Список доступных тестов
        """

        test_methods = [method for method in dir(self) if
                        callable(getattr(self, method)) and method.startswith("test")]
        return test_methods
    
    def generate_test_result(self):
        """Функция генерируют отчет по результатам тестирования
        """        
        self.file.close()
        

    
    def start_tests(self, list_of_tests: list) -> None:
        """Функция запускает заданные тесты

        Args:
            list_of_tests (list): Список тестов, которые необходимо запустить.
        """
        for method_name in list_of_tests:
            method = getattr(self, method_name)
            method()
    
    def get_method_description(self, method_name: str) -> str:
        """Метод для получение описания теста по названию теста

         Args:
             method_name(str) - название теста
        Returns:
             str - описание теста
         """
        method = getattr(self, method_name, None)
        if method is not None:
            return method.__doc__
        else:
            return "Метод не найден."

        

class QA_QC_GIS_first():
    def __init__(self, filename="file.txt", las_path = '../Данные/qaqc/ГИС/9281PL.las',) -> None:
        """Тесты первого порядка для каротажей ГИС.

        Args:
            filename (str, optional): Название файла с результатами теста. Defaults to "file.txt".
            las_path (str, optional): Путь к las файлу с каротажами ГИС. Defaults to '../Данные/qaqc/ГИС/9281PL.las'.
        """
        self.filename = filename
        self.file = open(self.filename, "w")

        self.las = lasio.read(las_path).df()
        self.data = lasio.read(las_path)
        self.gis, self.unit = self.gis_preparing(top = self.las.index[0], bottom = self.las.index[-1])

        

    def gis_preparing(self, top: float = 0, bottom: float = 3000) -> dict:
        """Функция, используя мнемоники, определяет, какие каротажи есть в .las файле. Обрезает каротажи по отбивкам кровли и подошвы пласта

        Args:
            top (float): Глубина верхней границы интересующего интервала\n
            bottom (float): Глубина нижней границы интересующего интервала\n

        Returns:
            dict: Словарь в формате key: Мнемоника каротажа, value: Каротаж.
        """     
        mnemonics = {'sp': ['SP', 'PS', 'ПС', 'СП', 'PS_1', 'PS_2'],
                    'gr': ['GR', 'GK', 'ГК', 'ECGR', 'GK_1', 'GK_2'],
                    'ds': ['DS', 'CALI', 'HCAL', 'CALIP', 'DCAV', 'DSN', 'DS_1', 'DS2'],
                    'mds': ['MCAL', 'MDS'],
                    'minv': ['МГЗ', 'MGZ'],
                    'mnor': ['МПЗ', 'MPZ'],
                    'mll': ['MLL', 'MBK', 'МБК', 'МКЗ', 'MSFL', 'RXOZ'],
                    'rhob': ['RHOB', 'PL', 'GGKP', 'RHOZ', 'DRHB', 'ROBB'],
                    'pef': ['PEF', 'PE', 'PEFZ', 'ZEFF'],
                    'dt': ['AK', 'DT', 'DTp', 'DTP', 'АК', 'AK_2', 'DTL', 'DTS', 'DTP1'],
                    'ild': ['IK', 'ILD', 'ИК', 'CILD', 'IKA', 'ILDA', 'IK_1', 'R27PC_46PH'],
                    'ildr': ['IKR', 'ILDR'],
                    'vikiz': ['ВИКИЗ', 'VIKIZ', 'F05', 'F07', 'F10', 'F14', 'F20', 'R05', 'R07', 'R10', 'R14', 'R20', 'С05', 'С07', 'С10', 'С14', 'С20', 'AT10', 'AT20', 'AT30', 'AT60', 'AT90', 'AF10', 'AF20', 'AF30', 'AF60',
                        'AF90', 'ILDVG1', 'ILDVG2', 'ILDVG3', 'ILDVG4', 'ILDVG5', 'ILDVR1', 'ILDVR2', 'ILDVR3', 'ILDVR4', 'ILDVR5', 'R1', 'R2', 'R3', 'R4', 'R5', 'VIK1', 'VIK2', 'VIK3', 'VIK4', 'VIK5', 'RO05', 'RO07',
                        'RO10', 'RO14', 'RO20', 'IK1', 'IK2', 'IK3', 'IK4', 'IK5', 'С07', 'С10', 'С14', 'С20', 'AT10', 'AT20', 'AT30', 'AT60', 'AT90', 'AF10', 'AF20', 'AF30', 'AF60', 'AF90', 'ILDVG1', 'ILDVG2',
                        'ILDVG3', 'ILDVG4', 'ILDVG5', 'ILDVR1', 'ILDVR2', 'ILDVR3', 'ILDVR4', 'ILDVR5', 'R1', 'R2', 'R3', 'R4', 'R5', 'VIK1', 'VIK2', 'VIK3', 'VIK4', 'VIK5', 'RO05', 'RO07', 'RO10', 'RO14', 'RO20',
                        'IK1', 'IK2', 'IK3', 'IK4', 'IK5'],
                    'ngr': ['NEUT', 'NGR', 'NGK', 'NGK_1'],
                    'nktd': ['NKTD', 'CFTC', 'NKT', 'NKTB', 'NKT_1', 'NKTB2'],
                    'nkts': ['NKTS', 'CNTC', 'NKTM', 'NKT_2'],
                    'w': ['NPHI','W','TNPH','NPLS','NPSS','TNPD','TNPL','TNPS','TNPH','TNPH_DOL','TNPH_LIM','TNPH_SAN'],
                    'depth': ['DEPT']}   
               
        data = self.las[(self.las.index > top) & (self.las.index < bottom)]
        units_list = [i.upper() for i in data]
        gis = {}
        units = {}
        for mnemonic in mnemonics.keys():
            for unit in units_list:
                if unit in  mnemonics.get(mnemonic):
                    gis[mnemonic] = np.array(data[unit])
                    units[mnemonic] = self.data.curves[unit].unit

        gis['depth'] = data.index.to_numpy()

        return gis, units

    def check_input(self, array, param_name, test_name) -> bool:
        """Функция для проверки входных данных

            Args:
                self.data (array[T]): входной массив для проверки данных
                param_name (str): Название параметра
                test_name (str): Название теста

            Returns:
                bool: результат выполнения теста
        """

        if not isinstance(array, np.ndarray):
            self.file.write(f"Тест {test_name} не запускался. Причина {param_name} не является массивом\n")
            return False
        if len(array) == 0:
            self.file.write(f"Тест {test_name} не запускался. Причина {param_name} пустой\n")
            return False
        for element in array:
            if not isinstance(element, (int, float)):
                self.file.write(
                    f"Тест {test_name} не запускался. Причина {param_name} содержит данные типа не int/float\n")
                return False
        return True
    

    def test_all(self) -> bool:
        """Проверка для всех каротажей, кроме SP, DS, MDS, RHOB, DT, PEF, DT, W, БК, ИК"""
        gis_for_another_test = ['sp', 'ds', 'mds', 'rhob', 'dt', 'pef', 'dt', 'w', 'bk', 'ild']
        for i in self.gis.keys():
            result = None
            if i not in gis_for_another_test and self.check_input(self.gis[i], self.gis[i], 'тест на физичность'):
                if all(x > 0 or x == -9999 or x == -46425 or x == -999.25 or np.isnan(x)  for x in self.gis[i]):
                    result = True
                else:
                    result = False
                self.file.write(f"Test for {i}: {result}\n")
        return result
    
    def test_sp(self) -> bool:
        """_summary_

        Returns:
            bool: _description_
        """        
        if 'sp' in self.gis.keys() and self.check_input(self.gis['sp'], self.gis['sp'], 'тест на физичность параметров'):
                if self.unit['sp'] == 'mV' or self.unit['sp'] == 'mv':
                    metric = 1
                    if all(-1500 < x*metric < 1500 or x == -9999 or x == -46425 or np.isnan(x)  for x in self.gis['sp']):
                        result = True
                    else:
                        result = False
                elif self.unit['sp'] == 'V' or self.unit['sp'] == 'v':
                    metric = 0.001
                    if all(-1500 < x*metric < 1500 or x == -9999 or x == -46425 or np.isnan(x)  for x in self.gis['sp']):
                        result = True
                    else:
                        result = False
                else:
                    result = False
                    self.file.write('Единица измерения  - ' + str(self.unit['sp'])  + '. ')
                self.file.write("Test 'SP': {}\n".format(result))
                return result
    
    def test_ds(self) -> bool:
        """_summary_

        Returns:
            bool: _description_
        """        
        if 'ds' in self.gis.keys() and self.check_input(self.gis['ds'], self.gis['ds'], 'тест на физичность параметров'):
            if self.unit['ds'] == 'cm' or self.unit['ds'] == 'см' or self.unit['ds'] == 'mm' or self.unit['ds'] == 'мм' or self.unit['ds'] == 'm' or self.unit['ds'] == 'м':
                if self.unit['ds'] == 'см' or self.unit['ds'] == 'cm':
                    mnoj = 100
                elif self.unit['ds'] == 'm' or self.unit['ds'] == 'м':
                    mnoj = 1
                elif self.unit['ds'] == 'mm' or self.unit['ds'] == 'мм':
                    mnoj = 1000
                if all(0.1*mnoj < x < 0.5*mnoj or x == -9999 or x == -46425 or x == -999.25 or np.isnan(x) for x in self.gis['ds']):
                    result = True
                else:
                    result = False
            else:
                result = False
                self.file.write('Единица измерения  - ' + str(self.unit['ds'])  + '. ')
            self.file.write("Test 'DS': {}\n".format(result))
            return result
    
    def test_mds(self) -> bool:
        """_summary_

        Returns:
            bool: _description_
        """        
        if 'mds' in self.gis.keys() and self.check_input(self.gis['mds'], self.gis['mds'], 'тест на физичность параметров'):
            if self.unit['mds'] == 'cm' or self.unit['mds'] == 'см' or self.unit['mds'] == 'mm' or self.unit['mds'] == 'мм' or self.unit['mds'] == 'm' or self.unit['mds'] == 'м':
                if self.unit['mds'] == 'см' or self.unit['mds'] == 'cm':
                    mnoj = 100
                elif self.unit['mds'] == 'm' or self.unit['mds'] == 'м':
                    mnoj = 1
                elif self.unit['mds'] == 'mm' or self.unit['mds'] == 'мм':
                    mnoj = 1000
                if all(0.1*mnoj < x < 0.5*mnoj or x == -9999 or x == -46425 or x == -999.25 or np.isnan(x) for x in self.gis['mds']):
                    result = True
                else:
                    result = False
            else:
                result = False
                self.file.write('Единица измерения  - ' + str(self.unit['w'])  + '. ')
            self.file.write("Test 'MDS': {}\n".format(result))
            return result
    
    def test_rhob(self) -> bool:
        """_summary_

        Returns:
            bool: _description_
        """        
        if 'rhob' in self.gis.keys() and self.check_input(self.gis['rhob'], self.gis['rhob'], 'тест на физичность параметров'):
            if self.unit['rhob'] == 'g/cc' or self.unit['rhob'] == 'g/cm3' or self.unit['rhob'] == 'кг/м3' or self.unit['rhob'] == 'kg/m3':
                if self.unit['rhob'] == 'g/cc' or self.unit['rhob'] == 'g/cm3':
                    mnoj = 1
                elif self.unit['rhob'] == 'kg/m3' or self.unit['rhob'] == 'кг/м3':
                    mnoj = 1000
                if all(1.5*mnoj < x < 3.5*mnoj or x == -9999 or x == -46425 or x == -999.25 or np.isnan(x) for x in self.gis['rhob']):
                    result = True
                else:
                    result = False
                    self.file.write('Value not in range')
            else:
                result = False
                self.file.write('Единица измерения: '+self.unit['rhob']  + '. ')
            self.file.write("Test 'RHOB': {}\n".format(result))
            return result
    
    def test_pef(self) -> bool:
        """_summary_

        Returns:
            bool: _description_
        """        
        if 'pef' in self.gis.keys() and self.check_input(self.gis['pef'], self.gis['pef'], 'тест на физичность параметров'):
            if self.unit['pef'] == 'b/e':
                if all(0 < x < 10 or x == -9999 or x == -46425 or x == -999.25 or np.isnan(x)for x in self.gis['pef']):
                    result = True
                else:
                    result = False
            else:
                result = False
                self.file.write('Единица измерения  - ' + str(self.unit['pef']) + '. ')
            self.file.write("Test 'PEF': {}\n".format(result))
            return result
    
    def test_dt(self) -> bool:
        """_summary_

        Returns:
            bool: _description_
        """        
        if 'dt' in self.gis.keys() and self.check_input(self.gis['dt'], self.gis['dt'], 'тест на физичность параметров'):
            if self.unit['dt'] == 'us/m' or self.unit['dt'] == 'us/f':
                if self.unit['dt'] == 'us/m':
                    mnoj=1
                elif self.unit['dt'] == 'us/f':
                    mnoj=0.3
                if all(100*mnoj < x < 800*mnoj or x == -9999 or x == -46425 or x == -999.25 or np.isnan(x) for x in self.gis['dt']):
                    result = True
                else:
                    result = False
            else:
                result = False
                self.file.write('Единица измерения  - ' + str(self.unit['dt']) + '. ')
            self.file.write("Test 'DT': {}\n".format(result))
            return result
    
    def test_W(self) -> bool:
        """_summary_

        Returns:
            bool: _description_
        """        
        if 'w' in self.gis.keys() and self.check_input(self.gis['w'], self.gis['w'], 'тест на физичность параметров'):
            if self.unit['w'] == 'v/v' or self.unit['w'] == '%':
                if self.unit['w'] == 'v/v':
                    mnoj = 1 
                elif self.unit['w'] == '%':
                    mnoj = 100
                if all(-0.15*mnoj < x < 1*mnoj or x == -9999 or x == -46425 or x == -999.25 or np.isnan(x) for x in self.gis['w']):
                    result = True
                else:
                    result = False
            else:
                result = False
                self.file.write('Единица измерения  - ' + str(self.unit['w']) + '. ')
            self.file.write("Test 'W': {}\n".format(result))
            return result
    
    def test_bk(self) -> bool:
        """_summary_

        Returns:
            bool: _description_
        """                
        if 'bk' in self.gis.keys() and self.check_input(self.gis['bk'], self.gis['bk'], 'тест на физичность параметров'):
            if self.unit['bk'] == 'Omm':
                if all( x*metric > 0.3 or x == -9999 or x == -46425 or x == -999.25 or np.isnan(x) for x in self.gis['bk']):
                    result = True
                else:
                    result = False
            else: 
                result = False
                self.file.write('Единица измерения  - ' + str(self.unit['bk']) + '. ')
            self.file.write("Test 'BK': {}\n".format(result))
            return result
        
    def test_ik(self) -> bool:
        """_summary_

        Returns:
            bool: _description_
        """        
        if 'ild' in self.gis.keys() and self.check_input(self.gis['ild'], self.gis['ild'], 'тест на физичность параметров'):
            if self.unit['ild'] == 'ohmm' or self.unit['ild'] == 'ohm.m':
                if all( x*metric > 0.2 or x == -9999 or x == -46425 or x == -999.25 or np.isnan(x) for x in self.gis['ild']):
                    result = True
                else:
                    result = False
            else:
                result = False
                self.file.write('Единица измерения  - ' + str(self.unit['ild']) + '. ')
            self.file.write("Test 'ILD': {}\n".format(result))
            return result
    
    def get_list_of_tests(self) -> list:
        """Возвращает список доступных тестов

        Returns:
            list: список тестов
        """

        test_methods = [method for method in dir(self) if
                        callable(getattr(self, method)) and method.startswith("test")]
        return test_methods

    def start_tests(self, list_of_tests: list) -> None:
        """Запускает тесты, переданные в атрибуте

        Args:
            list_of_tests (list): _description_
        """
        for method_name in list_of_tests:
            method = getattr(self, method_name)
            method()
        
        

    def generate_test_report(self) -> str:
        """Генерирет отчет по результатам тестирования

        Returns:
            str: _description_
        """
        self.file.close()