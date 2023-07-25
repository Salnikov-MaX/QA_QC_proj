import numpy as np

import lasio
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import scipy.stats as stats


class QA_QC_GIS_second:
    def __init__(self, las_path: str, bounds: tuple, poro_open: np.ndarray=None, perm: np.ndarray=None, poro_eff: np.ndarray=None, lithology: np.ndarray=None, depth: np.ndarray=None, **mnemonics_add) -> None:
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
        self.las_depth_unit = self.las.curves['dept'].unit
        self.las = self.las.df()
        self.mnemonics_add = mnemonics_add
        self.gis = self.gis_preparing(top = bounds[0], bottom = bounds[1])
        self.depth = depth
        self.poro_open = poro_open
        self.perm = perm
        self.poro_eff = poro_eff
        self.lithology = lithology
        self.bounds = bounds

    def check_input(self, array, param_name, test_name) -> bool:
        """Функция для проверки входных данных

            Args:
                self.data (array[T]): входной массив для проверки данных

            Returns:
                bool: результат выполнения теста
        """

        if not isinstance(array, np.ndarray):
            self.file.write(f"Тест {test_name} не запускался. Причина {param_name} не является массивом\n")
            return False
        if len(array) == 0:
            self.file.write(f"Тест {test_name} не запускался. Причина {param_name} пустой\n")
            return False
        if False not in np.isnan(array): 
            self.file.write(f"Тест {test_name} некорректен. Причина: {param_name} состоит из NaN\n")
        for element in array:
            if not isinstance(element, (int, float)):
                self.file.write(
                    f"Тест {test_name} не запускался. Причина {param_name} содержит данные типа не int/float\n")
                return False
        return True


    def check_input_lithology(self, array, param_name, test_name) -> bool:
        """Функция для проверки входных данных

            Args:
                array (array[T]): входной массив для проверки данных

            Returns:
                bool: результат выполнения теста
        """

        if not isinstance(array, np.ndarray):
            self.file.write(f"Тест {test_name} не запускался. Причина {param_name} не является массивом\n")
            return False
        if len(array) == 0:
            self.file.write(f"Тест {test_name} не запускался. Причина {param_name} пустой\n")
            return False
        if False not in pd.isnull(array): 
            self.file.write(f"Тест {test_name} некорректен. Причина: {param_name} состоит из NaN\n")
            return False
        for element in array:
            if not isinstance(element, (str)):
                self.file.write(
                    f"Тест {test_name} не запускался. Причина {param_name} содержит данные типа не str\n")
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
        for i in self.mnemonics_add.keys():
            mnemonics[i].append(self.mnemonics_add[i])   
        data = self.las[(self.las.index > top) & (self.las.index < bottom)]
        units_list = [i.upper() for i in data]
        gis = {}

        for mnemonic in mnemonics.keys():
            for unit in units_list:
                if unit in mnemonics.get(mnemonic):
                    gis[mnemonic] = np.array(data[unit])

        gis['depth'] = data.index.to_numpy()

        if gis['depth'][1] - gis['depth'][0] != 0.01:
            start = round(gis['depth'][0], 2)
            stop = round(gis['depth'][-1], 2)
            step = round(gis['depth'][1] - gis['depth'][0],2)
            interp_linspace = np.linspace(start, stop, round((step)/0.01*len(gis['depth']))-round((step/0.01-1)))
            for i in gis.keys():
                if i != 'depth':
                    gis[i] = np.interp(interp_linspace, gis['depth'], gis[i])
        gis['depth'] = np.array([round(x,2) for x in interp_linspace])


        return gis
    
    
    def kernpreproc(self) -> np.ndarray:
        """
            Функция заменяет буквенное наименование литологии по керну на численное
        """
        kern_lithology = []

        for i in self.lithology:
            if i[0] == 'П':
                kern_lithology.append(1)
            elif i[0:2] == 'Ал':
                kern_lithology.append(2)
            elif i[0:2] == 'Ар' or i[0:2] == 'Гл': 
                kern_lithology.append(3)
            else:
                kern_lithology.append(0)

        return np.array(kern_lithology)
    
    def test_lithology(self, siltmin: float = 0.4, siltmax: float = 0.7, sandmin: float = 0, sandmax: float = 0.4, argillitemin: float = 0.7, argillitemax: float = 1, distance: float=50) -> None:
        """Тест проверяет качество увязки литологии по керновым данным и литологии по SP или GR каротажам.

        Args:
            siltmin (float, optional): Левая граница интервала амплитуд ПС и ГК, соответствующая алевролитам. Defaults to 0.4.\n
            siltmax (float, optional): Правая граница интервала амплитуд ПС и ГК, соответствующая алевролитам. Defaults to 0.7.\n
            sandmin (float, optional): Левая граница интервала амплитуд ПС и ГК, соответствующая песчанику. Defaults to 0.\n
            sandmax (float, optional): Правая граница интервала амплитуд ПС и ГК, соответствующая песчанику. Defaults to 0.4.\n
            argillitemin (float, optional): Левая граница интервала амплитуд ПС и ГК, соответствующая аргиллиту. Defaults to 0.7.\n
            argillitemax (float, optional): Правая граница интервала амплитуд ПС и ГК, соответствующая аргиллиту. Defaults to 1.\n
            distance (float, optional): Для нормализации значений каротажа необходимо взять значения выше и ниже рассматриваемого пласта, данный атрибут показывает, на сколько выше и ниже. Defaults to 50.
        """      
        if ('sp' or 'gr' in self.gis.keys()) and self.check_input_lithology(self.lithology, 'Литология по керну', 'Увязка керн и ГИС по литологии') and self.check_input(self.depth, 'Глубина отбора керна', 'Увязка керн и ГИС по литологии'):
             
            kern_lithology = self.kernpreproc()
            gis_lithology = self.gis_preparing(top=self.bounds[0]-distance, bottom=self.bounds[1]+distance)
            scaler = MinMaxScaler()

            if 'sp' in self.gis.keys():
                log_for_lithology = scaler.fit_transform(gis_lithology['sp'].reshape(-1,1))
            elif 'gr' in self.gis.keys(): 
                log_for_lithology = scaler.fit_transform(gis_lithology['gr'].reshape(-1,1))
            
            depth_gis = gis_lithology['depth']
            litho = []
            count_matches = 0
            count = 0
            grforplot = []
            transparency = []

            for i in range(len(self.depth)):
                index = np.where(depth_gis == round(self.depth[i], 1))[0][0]
                if self.depth[i] > 0 and log_for_lithology[index] >= 0:
                    if round(self.depth[i], 1) < depth_gis[-1]:
                            count = count + 1
                            if sandmin <= log_for_lithology[index] < sandmax and kern_lithology[i] == 1:
                                count_matches = count_matches + 1
                                litho.append(1)
                                grforplot.append(log_for_lithology[index])
                                transparency.append(0.001)
                            elif siltmin < log_for_lithology[index] < siltmax and kern_lithology[i] == 2:
                                count_matches = count_matches + 1
                                litho.append(2)
                                grforplot.append(log_for_lithology[index])
                                transparency.append(0.001)
                            elif argillitemin < log_for_lithology[index] <= argillitemax and kern_lithology[i] == 3:
                                count_matches = count_matches + 1
                                litho.append(3)
                                grforplot.append(log_for_lithology[index])
                                transparency.append(0.001)
                            else:
                                litho.append(0)
                                grforplot.append(log_for_lithology[index])
                                transparency.append(1)
            
            self.lithology_test_visualization(grforplot, transparency, kern_lithology, count_matches, count)
    
    
    def lithology_test_visualization(self, grforplot: list, prozr: list, kern_lithology: list, count: int, count1: int):
            """Функция визуализирует результаты теста увязки литологии по керну и по ГИС. Отображает каротаж ГИС и отмечает на нем точки, в которых литология не увязана. Также отображает литологию по керну. Атрибуты функции указывать не нужно

            Args:
                grforplot (list): \n
                prozr (list): _description_\n
                kern_lithology (list): _description_\n
                count (int): _description_\n
                count1 (int): _description_\n
            """          
            print('Тестирование качества увязки литологии по ГИС и литологии по керну')
            print('')
            print('Процент совпавших литотипов по ГИС и по керну равен ', str(count/count1*100), '%')
            self.file.write('Тестирование качества увязки литологии по ГИС и литологии по керну.\nПроцент совпавших литотипов по ГИС и по керну равен ' + str(count/count1*100) + ' %.\n')
            print('Оранжевыми точками отмечены глубины, в которых литология не увязана. В литологии по керну: песчаник - ф, алевролит - з, аргиллит/глина - ж')

            
            plt.figure(figsize=(4,10))
            plt.subplot(1,2,1)

            if 'sp' in self.gis.keys():
                plt.title('SP')
            else:
                plt.title('GR')
            
            plt.plot(grforplot, np.linspace(len(grforplot)-1, 0, len(grforplot)))
            plt.scatter(grforplot, np.linspace(len(grforplot)-1, 0, len(grforplot)), c = 'orange', alpha = prozr)
            plt.ylim(0,501)
            plt.xticks([])
            plt.yticks([])
            plt.subplot(1,2,2)
            plt.title('Литология по керну')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(np.array(kern_lithology).reshape(-1, 1),  aspect='auto')
            plt.show()

    def properties(self, poro_model = None, poroeff_model = None, perm_model = None, gis_type = 'rhob' ) -> pd.core.frame.DataFrame:
        """
            Данная функция создает РИГИС пористости, эффективной пористости и проницаемости
        Args:
            poro_model (function): петрофизическая модель пористости,\n
            poroeff_model (function): петрофизическая модель эффективной пористости,\n
            perm_model(function): петрофизическая модель проницаемости,\n
            gis_type (str): мнемоника каротажа ГИС, по которой построена модель пористости.\n 
        Returns:\n
            DataFrame: таблица  с РИГИСами пористости, эффективной пористости и проницаемости.
        """
        
        if not poro_model:
            poro_model = self.poro_model

        if not poroeff_model:
            poroeff_model = self.poroeff_model

        if not perm_model:
            perm_model = self.perm_model

        rigis = pd.DataFrame(columns=['poro', 'poroeff', 'perm', 'depth'])
        depth = self.gis['depth']
        gis_for_properties = self.gis[gis_type]
        kern_poro = []
        rigis_poro = []
        
        for i in range(len(self.depth)):
            if self.depth[i] > 0:
                    depthindex = np.where(depth == round(self.depth[i], 2))[0][0]
                    if gis_for_properties[depthindex] > 0 and self.poro_open[i] > 0:
                        o = len(rigis['poro'])
                        rigis.at[o, 'poro'] = poro_model(gis_for_properties[depthindex])
                        rigis.at[o, 'depth'] = depth[depthindex]
                        kern_poro.append(self.poro_open[i])
                        rigis_poro.append(float(poro_model(gis_for_properties[depthindex])))
                        rigis.at[o, 'poroeff'] = poroeff_model(rigis.loc[o]['poro'])
                        rigis.at[o, 'perm'] = perm_model(rigis.loc[o]['poroeff'])

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
            poroeff_koeff2 (float): Свободный член уравнения.
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

    def test_properties(self, poro_model = None, poroeff_model = None, perm_model = None, gis_type = 'rhob') -> None:
        """ 
            Тест нацелен на проверку качества увязки пористости, эффективной пористости и проницаемости по РИГИС с этими же данными по керну
        Args:
            poro_model (function): петрофизическая модель пористости,\n
            poroeff_model (function): петрофизическая модель эффективной пористости,\n
            perm_model(function): петрофизическая модель проницаемости,\n
            gis_type (str): мнемоника каротажа ГИС, по которой построена модель пористости.\n 
        """
        
        
        if self.check_input(self.depth, 'Глубина отбора керна', 'увязка керна и ГИС по свойствам') and self.check_input(self.poro_open, 'Открытая пористость по керну', 'Увязка керна и ГИС по открытой пористости') and \
            self.check_input(self.poro_eff, 'Эффективная пористость по керну', 'увязка керна и ГИС по эффективной пористости') and self.check_input(self.perm, 'Проницаемость по керну', 'Увязка керна и ГИС по проницаемости') and gis_type in self.gis.keys():
            
            rigis, kern_poro, rigis_poro = self.properties(poro_model = poro_model, poroeff_model = poroeff_model, perm_model = perm_model, gis_type = gis_type)

            if len(rigis.values) > 0:
                kern_poroeff = []
                rigis_poroeff = []

                for i in range(len(self.poro_eff)):
                    if self.poro_eff[i] > 0 and round(self.depth[i], 2) in list(self.rigis['depth']):
                        rigis_poroeff.append(float(rigis.loc[np.where(rigis['depth'] == round(self.depth[i], 2))[0][0]]['poroeff']))
                        kern_poroeff.append(self.poro_eff[i])
                
                kern_perm = []
                rigis_perm = []

                for i in range(len(self.perm)):
                    if self.perm[i] > 0 and round(self.depth[i], 2) in list(rigis['depth']):
                        rigis_perm.append(float(rigis.loc[np.where(rigis['depth'] == round(self.depth[i], 2))[0][0]]['perm']))
                        kern_perm.append(self.perm[i])
                
            self.properties_visualization(kern_poro, rigis_poro, kern_poroeff, rigis_poroeff, kern_perm, rigis_perm)
            
            

    def properties_visualization(self, kern_poro: list, rigis_poro: list, kern_poroeff: list, rigis_poroeff: list, kern_perm: list, rigis_perm: list) -> None :
        """ Функция визуализирует результаты теста на увязку пористости, эффективной пористости и проницаемости по керну и по ГИС.

        Args:
            kern_poro (_type_): _description_\n
            rigis_poro (_type_): _description_\n
            kern_poroeff (_type_): _description_\n
            rigis_poroeff (_type_): _description_\n
            kern_perm (_type_): _description_\n
            rigis_perm (_type_): _description_\n
        """
        print('')
        print('Тестирование качества увязки открытой пористости, эффективной пористости и проницаемости по керну с этими же свойствами по РИГИС')
        print('')
        self.file.write('Тестирование качества увязки открытой пористости, эффективной пористости и проницаемости по керну с этими же свойствами по РИГИС.\n')

        if list(kern_poro):
            k_poro, b_poro, r_poro = self.trendline(rigis_poro, kern_poro)
            plt.scatter(rigis_poro, kern_poro)
            plt.plot([0, sorted(rigis_poro)[-1]], [b_poro, k_poro*list(sorted(rigis_poro))[-1]+b_poro], c='orange')
            plt.xlabel('Пористость по керну, %')
            plt.ylabel('РИГИС пористости, %')
            plt.title('Коэффициент k = ' + str(k_poro) + ', коэффициент R^2 = ' + str(r_poro))
            plt.show()
            self.file.write('Пористость: Коэффициент k = ' + str(k_poro) + ', коэффициент R^2 = ' + str(r_poro) + '\n')

        else:
            print('В файле с керном нет данных по пористости')
            self.file.write('В файле с керном нет данных по пористости\n')

        if list(kern_poroeff):
            k_poroeff, b_poroeff, r_poroeff = self.trendline(rigis_poroeff, kern_poroeff)
            plt.scatter(rigis_poroeff, kern_poroeff)
            plt.plot([0, sorted(rigis_poroeff)[-1]], [b_poroeff, k_poroeff*sorted(rigis_poroeff)[-1]+b_poroeff], c='orange')
            plt.xlabel('РИГИС эффективной пористости, %')
            plt.ylabel('Эффективная пористость по керну, %')
            plt.title('Коэффициент k = ', str(k_poroeff), ', коэффициент R^2 = ', str(r_poroeff))
            plt.show()
            self.file.write('Эффективная пористость: Коэффициент k = ' + str(k_poro) + ', коэффициент R^2 = ' + str(r_poro) + '\n')

        else:
            print('')
            print('В файле с керном нет таких глубин, на которых были бы данные по эффективной пористости и по открытой пористости')
            self.file.write('В файле с керном нет таких глубин, на которых были бы данные по эффективной пористости и по открытой пористости\n')
            print('')
            

        if list(kern_perm):
            k_perm, b_perm, r_perm = self.trendline(rigis_perm, kern_perm)
            plt.scatter(rigis_perm, kern_perm)
            plt.plot([0, sorted(rigis_perm)[-1]], [b_perm, k_perm*sorted(rigis_perm)[-1]+b_perm], c='orange')
            plt.xlabel('РИГИС проницаемости, мД')
            plt.ylabel('Проницаемость по керну, мД')
            plt.title('Коэффициент k = ' + str(k_perm) + ', коэффициент R^2 = ' + str(r_perm))
            plt.show()
            self.file.write('Проницаемость: Коэффициент k = ' + str(k_perm) + ', коэффициент R^2 = ' + str(r_perm) + '\n')
            
        else:
            print('В файле с керном нет таких глубин, на которых были бы данные по открытой пористости и по проницаемости')
            self.file.write('В файле с керном нет таких глубин, на которых были бы данные по открытой пористости и по проницаемости.\n')
        
        

        
    
    def test_skipped_gis(self, delta: float=0.5) -> None:
        """Тест направлен на поиск пропусков в записи ГИС, и, в случае интервала пропусков меньше delta м, их заполнения интерполяцией.

        Args:
            delta (float, optional): Атрибут задает максимальный интервал в метрах, в пределах которого пропуски будут заполняться интерполяцией . Defaults to 0.5.
        """
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
        units_list = [i.upper() for i in self.las]
            
        repeat = {}

        for i in  units_list:
            if i+'1' in units_list:
                repeat[i] = i+'1'
            elif i+'2' in units_list:
                repeat[i] = i+'2'
            elif i+'_1' in units_list:
                repeat[i] = i+'_1'
            elif i+'_2' in units_list:
                repeat[i] = i+'_2'

        if repeat:
            for i in repeat.keys():
                gis = []
                for o in len(self.gis['depth']):
                    if not np.isnan(self.gis[i][o]) and not np.isnan(self.gis[repeat.get(i)][o]):
                        gis.append(np.average([self.gis[i][o],self.gis[repeat.get(i)][o]]))
                    elif not np.isnan(self.gis[i][o]):
                        gis.append(self.gis[i][o])
                    elif not np.isnan(self.gis[repeat.get(i)][o]):
                        gis.append(self.gis[repeat.get(i)][o])
                    else:
                        gis.append(np.nan)
                self.gis[i] = gis
        
            print('Перекрытие интервалов есть в каротажах: ')
            self.file.write('Перекрытие интервалов есть в каротажах: \n')
            for keys, values in repeat.items():
                print(keys, values)
                self.file.write(keys, values)
        else:
            print('Перекрытия интервалов в данных нет')
            self.file.write('Перекрытия интервалов в данных нет\n')
        

    def test_max_value_gis(self) -> None:
        """
        Тестирование на сходство максимальных отметок глубин по отбивкам и по ГИС
        """
        depth = self.las.index
        print('')
        print('Тестирование на сходство максимальных отметок глубин по отбивкам и по ГИС')
        print('')
        if depth[-1] <= self.bounds[1]:
            print('Максимальная глубина по отбивке ' + str(self.bounds[1]) + ', максимальная глубина по ГИС ' + str(depth[-1]) + str('. Тест не пройден'))
            self.file.write('Тестирование на сходство максимальных отметок глубин по отбивкам и по ГИС.\nМаксимальная глубина по отбивке ' + str(self.bounds[1]) + ', максимальная глубина по ГИС ' + str(depth[-1]) + str('. Тест не пройден\n'))
        else:
            print('Максимальная глубина по отбивке ' + str(self.bounds[1]) + ', максимальная глубина по ГИС ' + str(depth[-1]) + str('. Тест пройден'))
            self.file.write('Тестирование на сходство максимальных отметок глубин по отбивкам и по ГИС. \nМаксимальная глубина по отбивке ' + str(self.bounds[1]) + ', максимальная глубина по ГИС ' + str(depth[-1]) + str('. Тест пройден\n'))
            

    def test_saturation(self, Archi_model= None, J_model = None, gis_type1: str = 'ild', Pc: float= None, sigma: float= None, tetta: float= None, poro_model = None, poroeff_model = None, perm_model = None, gis_type: str= 'rhob'):
        """Тест направлен на проверку качества увязки моделей водонасыщенности по J функции, по Арчи и по ОФП между собой.

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
        if Pc and sigma and tetta:

            rigis = self.properties(poro_model = poro_model, poroeff_model = poroeff_model, perm_model = perm_model, gis_type = gis_type)
            
            if not Archi_model:
                Archi_model = self.Archi_model
            
            if not J_model:
                J_model = self.J_function
            
            Archi = []
            J_func = []

            for i in len(rigis.values()):
                porosity = rigis.loc[i]['poro']
                permeability = rigis.loc[i]['perm']
                Archi.append(Archi_model(self.gis[gis_type1][np.where(self.gis['depth'] == rigis.loc[i]['depth'])], porosity))
                J_func.append(J_model(Pc*(permeability/porosity)**0.5/(sigma*np.cos(tetta))))

            
            print('')
            print('Тестирование качества увязки открытой пористости, эффективной пористости и проницаемости по керну с этими же свойствами по РИГИС')
            print('')

            if list(Archi) and list(J_func):
                plt.scatter(Archi, J_func)
                k, b, r = self.trendline(Archi, J_func)
                plt.plot([0, sorted(Archi)[-1]], [b, k*list(sorted(Archi))[-1]+b], c='orange')
                plt.xlabel('Пористость по керну, %')
                plt.ylabel('Водонасыщенность, д.е.')
                plt.title('Коэффициент k = ' + str(k) + ', коэффициент R^2 = ' + str(r))
                plt.show()
                
            else:
                print('В файле с керном нет данных по пористости')

            self.saturation_report(np.array(Archi), np.array(J_func))
            

    def saturation_report(self, Archi: np.ndarray, J_func: np.ndarray) -> None:
            """ Результат тестирования качества увязки моделей водонасыщенности.

            Args:
                Archi (_type_): _description_\n
                J_func (_type_): _description_\n
            """            
            print('')
            print('Тестирование качества увязки моделей водонасыщенность')
            print('')

            if list(Archi) and list(J_func):
                plt.scatter(Archi, J_func)
                k, b, r = self.trendline(Archi, J_func)
                plt.plot([0, sorted(Archi)[-1]], [b, k*list(sorted(Archi))[-1]+b], c='orange')
                plt.xlabel('Пористость по керну')
                plt.ylabel('РИГИС пористости')
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
    

    def test_units(self, depth_unit: str= 'м'):
        """ Тест направлен на проверку соответствия единиц измерения глубины по ГИС и по керну

        Args:
            depth_unit (str): Единица измерения глубины по керну.
        """
        if depth_unit:
            if (self.las_depth_unit == 'м' or 'm') and (depth_unit == 'm' or 'м'):
                self.file.write('В ГИС и в керне глубина измеряется в метрах')
                print('В ГИС и в керне глубина измеряется в метрах')
            elif (self.las_depth_unit == 'фут' or 'ft') and (depth_unit == 'фут' or 'ft'):
                self.file.write('В ГИС и в керне глубина измеряется в футах')
                print('В ГИС и в керне глубина измеряется в футах')
            elif self.las_depth_unit == depth_unit:
                self.file.write('В ГИС и в керне глубина измеряется в ' + depth_unit)
                print('В ГИС и в керне глубина измеряется в ' + depth_unit)
            else:
                self.file.write('В ГИС глубина измеряется в ' + self.las_depth_unit + ', в керне глубина измеряется в ' + depth_unit)
                print('В ГИС глубина измеряется в ' + self.las_depth_unit + ', в керне глубина измеряется в ' + depth_unit)

    def get_list_of_tests(self) -> list:
        """Функция выводит список доступных тестов.

        Returns:
            list: Список доступных тестов
        """

        test_methods = [method for method in dir(self) if
                        callable(getattr(self, method)) and method.startswith("test")]
        return test_methods
    
    def generate_test_result(self):
        self.file.close()
        

    
    def start_tests(self, list_of_tests: list) -> None:
        """Функция запускает заданные тесты

        Args:
            list_of_tests (list): Список тестов, которые необходимо запустить.
        """
        for method_name in list_of_tests:
            method = getattr(self, method_name)
            method()

        

class QA_QC_GIS_first():
    def __init__(self, filename="file.txt", las_path = '../Данные/qaqc/ГИС/9281PL.las',) -> None:
        """_summary_

        Args:
            data (str): _description_
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
        units['depth'] = self.data.curves['DEPT'].unit

        return gis, units

    def check_input(self, array, param_name, test_name) -> bool:
        """Функция для проверки входных данных для тестов первого порядка

            Args:
                self.data (array[T]): входной массив для проверки данных

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
                print(result)
        return result
    
    def test_sp(self) -> bool:
        if 'sp' in self.gis.keys() and self.check_input(self.gis['sp'], self.gis['sp'], 'тест на физичность параметров'):
                if self.unit['sp'] == 'mV':
                    metric = 1
                    if all(-1500 < x*metric < 1500 or x == -9999 or x == -46425 or np.isnan(x)  for x in self.gis['sp']):
                        result = True
                    else:
                        result = False
                    self.file.write("Test 'SP': {}\n".format(result))
                elif self.unit['sp'] == 'V':
                    metric = 0.001
                    if all(-1500 < x*metric < 1500 or x == -9999 or x == -46425 or np.isnan(x)  for x in self.gis['sp']):
                        result = True
                    else:
                        result = False
                    self.file.write("Test 'SP': {}\n".format(result))
                else:
                    result = False
                    self.file.write("Test 'SP': {}\n".format(result) + ', так как единица измерения ' + self.unit['sp'])
                return result
    
    def test_ds(self) -> bool:
        if 'ds' in self.gis.keys() and self.check_input(self.gis['ds'], self.gis['ds'], 'тест на физичность параметров'):
            if self.unit['ds'] == 'm' or 'м':
                if all(0.1 < x < 0.5 or x == -9999 or x == -46425 or x == -999.25 or np.isnan(x) for x in self.gis['ds']):
                    result = True
                else:
                    result = False
            else:
                result = False
            self.file.write("Test 'DS': {}\n".format(result))
            return result
    
    def test_mds(self) -> bool:
        if 'mds' in self.gis.keys() and self.check_input(self.gis['mds'], self.gis['mds'], 'тест на физичность параметров'):
            if self.unit['mds'] == 'm' or 'м':
                if all(0.1 < x < 0.5 or x == -9999 or x == -46425 or x == -999.25 or np.isnan(x) for x in self.gis['mds']):
                    result = True
                else:
                    result = False
            else:
                result = False
            self.file.write("Test 'MDS': {}\n".format(result))
            return result
    
    def test_rhob(self) -> bool:
        if 'rhob' in self.gis.keys() and self.check_input(self.gis['rhob'], self.gis['rhob'], 'тест на физичность параметров'):
            if self.unit['rhob'] == 'g/cc' or self.unit['rhob'] == 'g/cm3':
                if all(1.5 < x < 3.5 or x == -9999 or x == -46425 or x == -999.25 or np.isnan(x) for x in self.gis['rhob']):
                    result = True
                else:
                    result = False
            else:
                result = False
            self.file.write("Test 'RHOB': {}\n".format(result))
            return result
    
    def test_pef(self) -> bool:
        if 'pef' in self.gis.keys() and self.check_input(self.gis['pef'], self.gis['pef'], 'тест на физичность параметров'):
            if self.unit['pef'] == 'b/e':
                if all(0 < x < 10 or x == -9999 or x == -46425 or x == -999.25 or np.isnan(x)for x in self.gis['pef']):
                    result = True
                else:
                    result = False
            else:
                result = False
            self.file.write("Test 'PEF': {}\n".format(result))
            return result
    
    def test_dt(self) -> bool:
        if 'dt' in self.gis.keys() and self.check_input(self.gis['dt'], self.gis['dt'], 'тест на физичность параметров'):
            if self.unit['dt'] == 'us/m':
                if all(100 < x < 800 or x == -9999 or x == -46425 or x == -999.25 or np.isnan(x) for x in self.gis['dt']):
                    result = True
                else:
                    result = False
            else:
                result = False
            self.file.write("Test 'DT': {}\n".format(result))
            return result
    
    def test_W(self) -> bool:
        if 'w' in self.gis.keys() and self.check_input(self.gis['w'], self.gis['w'], 'тест на физичность параметров'):
            if self.unit['w'] == 'v/v':
                if all(-0.15 < x < 1 or x == -9999 or x == -46425 or x == -999.25 or np.isnan(x) for x in self.gis['w']):
                    result = True
                else:
                    result = False
            else:
                result = False
            self.file.write("Test 'W': {}\n".format(result))
            return result
    
    def test_bk(self, metric=1) -> bool:
        """
        Args:
            metric (int): По умолчанию принимается единица измерения Omm, но если в данных она другая, то в гиперпараметре нужно указать множитель, переводящий в Omm 
            
        """
        if 'bk' in self.gis.keys() and self.check_input(self.gis['bk'], self.gis['bk'], 'тест на физичность параметров'):
            if self.unit['bk'] == 'Omm':
                if all( x*metric > 0.3 or x == -9999 or x == -46425 or x == -999.25 or np.isnan(x) for x in self.gis['bk']):
                    result = True
                else:
                    result = False
            else: 
                result = False
            self.file.write("Test 'BK': {}\n".format(result))
            return result
        
    def test_ik(self, metric=1) -> bool:
        """
        Args:
            metric (int): По умолчанию принимается единица измерения mS/m, но если в данных она другая, то в гиперпараметре нужно указать множитель, переводящий в mS/m 
            
        """
        if 'ild' in self.gis.keys() and self.check_input(self.gis['ild'], self.gis['ild'], 'тест на физичность параметров'):
            if self.unit['ild'] == 'ohmm' or self.unit['ild'] == 'ohm.m':
                if all( x*metric > 0.2 or x == -9999 or x == -46425 or x == -999.25 or np.isnan(x) for x in self.gis['ild']):
                    result = True
                else:
                    result = False
            else:
                result = False
            self.file.write("Test 'ILD': {}\n".format(result))
            return result
    
    def get_list_of_tests(self) -> list:
        """_summary_

        Returns:
            list: _description_
        """

        test_methods = [method for method in dir(self) if
                        callable(getattr(self, method)) and method.startswith("test")]
        return test_methods

    def start_tests(self, list_of_tests: list) -> None:
        """_summary_

        Args:
            list_of_tests (list): _description_
        """
        for method_name in list_of_tests:
            method = getattr(self, method_name)
            method()
        
        

    def generate_test_report(self) -> str:
        """_summary_

        Returns:
            str: _description_
        """
        self.file.close()