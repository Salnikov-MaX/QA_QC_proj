import numpy as np
import ruptures as rpt
import os

from matplotlib import pyplot as plt
from qa_qc_lib.tests.base_test import QA_QC_main
from qa_qc_lib.tools.math_tools import Anomaly_Search_Stat_Methods, sameSign
from qa_qc_lib.tests.wells.wells_nodes import Nodes_wells_data

class QA_QC_wells(QA_QC_main):
    def __init__(self, nodes_obj: Nodes_wells_data, folder_report: str):
        """
        Класс с тестами первого и второго порядка над историческими данными по скважинам

        Args:
            nodes_obj (Nodes_historical_wells_data): обьект, содержащий узлы с историческими данными по скважинам. \n
            folder_report (str): путь до папки с отчетамию \n
        """
        super().__init__()
        self.folder_report = folder_report
        self.nodes_obj = nodes_obj
        self.types_of_nodes_for_test = {'test_monotony': list(self.nodes_obj.keywords_cumulative.keys()) + ['DATE'],
                                        'test_anomaly': list(self.nodes_obj.keywords_debit.keys()),
                                        'test_limit_0_1': list(self.nodes_obj.keywords_wefac.keys()),
                                        'test_LR_eq_sum_OR_and_WR': ['LPR', 'OPR', 'WPR'],
                                        'test_imbalance_trends_with_BHP': {'base': 'BHP', 'debit': ['GPR', 'LPR'],
                                                                           'injection': ['WIR', 'GIR']},
                                        'test_imbalance_anomaly': {'base': 'BHP', 'debit': ['OPR', 'WPR', 'GPR', 'LPR'],
                                                                   'injection': ['WIR', 'GIR']}}

        self.order_tests = {1: ['test_monotony', 'test_anomaly', 'test_limit_0_1'],
                            2: ['test_LR_eq_sum_OR_and_WR', 'test_imbalance_trends_with_BHP', 'test_imbalance_anomaly']}

        self.report_function = {'test_monotony': self.get_report_first_order_test,
                                'test_anomaly': self.get_report_first_order_test,
                                'test_limit_0_1': self.get_report_first_order_test,
                                'test_LR_eq_sum_OR_and_WR': self.get_report_test_LR_eq_sum_OR_and_WR,
                                'test_imbalance_trends_with_BHP': self.get_report_test_imbalance_trends_with_BHP,
                                'test_imbalance_anomaly': self.get_report_test_imbalance_anomaly}

        self.nodes_anomalies = self.init_nodes_anomalies()
        
    def init_nodes_anomalies(self):
        """Используется при инициализации класса, для первоначального заполнения сведений об аномалиях в каждом узле

        Required data:
            self.nodes_obj.wells (list): список скважин,\n
            self.nodes_obj.nodes_wells (dict):  словарь узлов скважин с временными рядами

        Returns:
			nodes_anomalies: dict, словарь с узлами скважин.
                {well_name (str): {name_node (str): value (np.zeros)}}
        """
        nodes_anomalies = {}
        for w in self.nodes_obj.wells:
            nodes = dict()
            for k, v in self.nodes_obj.nodes_wells[w].items():
                nodes[k] = np.zeros(len(v))
            nodes_anomalies[w] = nodes

        return nodes_anomalies

    def get_specification(self, result_mask: np.array, test_name: str, error_decr: str, well_name: str, node_names: list):

        """Возвращает спецификацию теста

        Required data:
            result_mask (np.array(int)): результат работы теста,\n
            test_name (str):  название теста
            error_decr (str): описание ошибки   
            well_name (str): имя скважины
            node name (list): список узлов, над которыми проводился тест   

        Returns:
			specification: {
						"result_mask" : np.array(int), 0 - значение правильное,\n
                                                      1 - значение ошибочное,\n
                        "time_scale" : np.array(datetime.date), шкала времени для временного ряда
                        "test_name" : str, название теста
                        "error_decr": text, описание ошибки   
                        "well_name": str, имя скважины
                        "node_names": list, список имен узлов                                               
					}
			}
        """

        return {"result_mask": result_mask,
                "time_scale": [str(d) for d in self.nodes_obj.time_scale],
                "test_name": test_name,
                "error_decr": error_decr,
                "well_name": well_name,
                "node_names": node_names}

    def test_limit_0_1(self, node: np.array, node_name: str, well_name: str, get_report=True) -> dict:
        """
        Метод проверяет коэф эксплуатации скважины на соответствие интервалу [0,1]

        Required data:
            node (np.array): временной ряд, по одному  из keywords_wefac показателю скважины\n
            node_name (str): имя типа узла (одно из keywords_wefac) \n
            well_name (str): имя скважины, которо принадлежит узел 

        Args:
            get_report (bool, optional): Определяет, нужно ли отображать отчет. Defaults to True.

        Returns:
			{
				"data_availability": bool (выполнялся или нет тест)
				"result": bool 
				"specification": {
						"result_mask" : np.array(int), 0 - значение временного ряда лежит в пределах интервала [0,1],\n
                                                       1 - значение временного ряда находится за пределами интервала [0,1],\n
                        "time_scale" : np.array(datetime.date), шкала времени для временного ряда
                        "test_name" : str, название теста
                        "error_decr": text, описание ошибки   
                        "well_name": str, имя скважины
                        "node_names": list, список имен узлов                                                
					}
			}
        """

        if not node_name in self.types_of_nodes_for_test['test_limit_0_1']:
            return {"data_availability": False}

        mask_0 = node < 0
        mask_1 = node > 1
        result_mask = (mask_0 + mask_1).astype(int)
        result = result_mask.sum() == 0

        if result:
            text = 'Значения ряда находяться в интервале от 0 до 1'
            report_text = self.generate_report_text(text, 1)
        else:
            text = 'Некоторые значения ряда находяться за пределами интервала от 0 до 1'
            report_text = self.generate_report_text(text, 0)

        self.update_report(report_text)
        specification = self.get_specification(result_mask, "test_limit_0_1", report_text, well_name, [node_name])

        if get_report:
            self.report_function['test_limit_0_1'](report_text, specification)

        return {"data_availability": True, "result": bool(result), "specification": specification}

    def test_monotony(self, node, node_name: str, well_name: str, get_report=True) -> dict:
        """
        Метод проверяет значения временного ряда на монотонное возрастание (каждое следующее значение больше предыдущего)

        Required data:
            self.nodes_obj (Nodes_historical_wells_data): обьект, содержащий узлы с историческими данными по скважинам\n
            node (np.array): временной ряд, по одному  из keywords_cumulative показателю скважины\n
            node_name (str): имя типа узла (одно из keywords_cumulative) \n
            well_name (str): имя скважины, которо принадлежит узел 

        Args:
            get_report (bool, optional): Определяет, нужно ли отображать отчет. Defaults to True.

        Returns:
            {
				"data_availability": bool (выполнялся или нет тест)
				"result": bool 
				"specification": {
						"result_mask" : np.array(int), 0 - значение временного ряда больше предыдущего,\n
                                                       1 - значение временного ряда меньше или равно предыдущему,\n
                        "time_scale" : np.array(datetime.date), шкала времени для временного ряда
                        "test_name" : str, название теста
                        "error_decr": text, описание ошибки   
                        "well_name": str, имя скважины
                        "node_name": str, имя узла                                                
					}
			}
        """
        if not node_name in self.types_of_nodes_for_test['test_monotony']:
            return {"data_availability": False}

        idx_start = self.nodes_obj.init_idx_ts_wells[well_name]
        result_mask = np.diff(node[idx_start:]) <= 0
        result_mask = np.insert(result_mask, 0, False)
        result_mask = np.append(node[:idx_start], result_mask)
        result = result_mask.sum() == 0

        if result:
            text = 'Значения ряда монотонно возрастают'
            report_text = self.generate_report_text(text, 1)
        else:
            text = 'Знаяения ряда не возрастают монотонно'
            report_text = self.generate_report_text(text, 0)

        self.update_report(report_text)
        specification = self.get_specification(result_mask, "test_monotony", report_text, well_name, [node_name])

        if get_report:
            self.report_function['test_monotony'](specification)

        return {"data_availability": True, "result": result, "specification": specification}

    def test_anomaly(self, node, node_name: str, well_name: str, get_report=True) -> dict:
        """
        Метод проверяет наличие аномалий типа "выброс" и "сдвиг" в передаваемом временном ряду.\n
        Если аномалии обнаружены, то они записываются в словарь self.nodes_anomalies дл я соответсвующей скважины и узла. 

        Required data:
            self.nodes_obj (Nodes_historical_wells_data): обьект, содержащий узлы с историческими данными по скважинам\n
            node (np.array): временной ряд, по одному  из Nodes_historical_wells показателю скважины\n
            node_name (str): имя типа узла (одно из Nodes_historical_wells) \n
            well_name (str): имя скважины, которо принадлежит узел 

        Args:
            get_report (bool, optional): Определяет, нужно ли отображать отчет. Defaults to True.

        Returns:
            {
				"data_availability": bool (выполнялся или нет тест)
				"result": bool 
				"specification": {
						"result_mask" : np.array(int), 0 - значение временного ряда больше предыдущего,\n
                                                       1 - значение временного ряда аномально,\n
                        "time_scale" : np.array(datetime.date), шкала времени для временного ряда
                        "test_name" : str, название теста
                        "error_decr": text, описание ошибки   
                        "well_name": str, имя скважины
                        "node_name": str, имя узла                                                
					}
			}
        """
        if not node_name in self.types_of_nodes_for_test['test_anomaly']:
            return {"data_availability": False}

        idx_start = self.nodes_obj.init_idx_ts_wells[well_name]
        result_mask = Anomaly_Search_Stat_Methods(node[idx_start:]).find_anomalies(shld=6, method=2,
                                                                                   threshold_fraction=0.2)
        result_mask = np.array(result_mask).astype(int)
        result_mask = np.append(np.zeros(idx_start), result_mask)
        self.nodes_anomalies[well_name][node_name] = result_mask
        result = result_mask.sum() == 0

        if result:
            text = 'В значениях ряда не наблюдаются аномалии'
            report_text = self.generate_report_text(text, 1)
        else:
            text = 'В знаяениях ряда присутствуют аномалии' 
            report_text = self.generate_report_text(text, 0)

        self.update_report(report_text)
        specification = self.get_specification(result_mask, "test_anomaly", report_text, well_name, [node_name])

        if get_report:
            self.report_function['test_anomaly'](specification)

        return {"data_availability": True, "result": result, "specification": specification}

    def test_LR_eq_sum_OR_and_WR(self, well_name: str, get_report=True) -> dict:
        """
        Метод проверяет каждое значения временного ряда LPR на равенство сумме значений временных рядов OPR и WPR \n
        (LPR = OPR + WPR) \n

        Required data:
            self.nodes_obj (Nodes_historical_wells_data): обьект, содержащий узлы с историческими данными по скважинам\n
            well_name (str): имя скважины, которо принадлежат узелы \n


        Args:
            get_report (bool, optional): Определяет, нужно ли отображать отчет. Defaults to True.

        Returns:
            {
				"data_availability": bool (выполнялся или нет тест)
				"result": bool 
				"specification": {
						"result_mask" : np.array(int), 0 - значение узла LPR равно сумме значений в узлах OPR и WPR,\n
                                                       1 - значение узла LPR не  авно сумме значений в узлах OPR и WPR,\n
                        "time_scale" : np.array(datetime.date), шкала времени для временного ряда
                        "test_name" : str, название теста
                        "error_decr": text, описание ошибки                                
                        "well_name": str, имя скважины
                        "node_names": list =  список имен узлов, над которыми проводился тест                                                
					}
			}
        """
        needed_nodes = self.types_of_nodes_for_test['test_LR_eq_sum_OR_and_WR']
        nodes_well = self.nodes_obj.nodes_wells[well_name]
        if len(set(nodes_well.keys()) & set(needed_nodes)) != 3:
            return {"data_availability": False}

        node_LPR = nodes_well[needed_nodes[0]]
        node_OPR = nodes_well[needed_nodes[1]]
        node_WPR = nodes_well[needed_nodes[2]]
        result_mask = (np.round(node_LPR, 1) != np.round(node_OPR + node_WPR, 1)).astype(int)
        result = result_mask.sum() == 0

        if result:
            text = 'значения  ряда LPR равны сумме значений рядов OPR и WPR'
            report_text = self.generate_report_text(text, 1)
        else:
            text = 'значения  ряда LPR не равны сумме значений рядов OPR и WPR'
            report_text = self.generate_report_text(text, 0)

        self.update_report(report_text)
        specification = self.get_specification(result_mask, "test_LR_eq_sum_OR_and_WR", report_text, well_name,
                                               needed_nodes)

        if get_report:
            self.report_function['test_LR_eq_sum_OR_and_WR'](report_text, specification)

        return {"data_availability": True, "result": result, "specification": specification}

    def test_imbalance_trends_with_BHP(self, well_name: str, get_report=True) -> dict:
        """
        Метод проверяет: при тренде на снижении ВНР должно наблюдаться рост LPR/GPR (и наоборот).\n

        Required data:
            self.nodes_obj (Nodes_historical_wells_data): обьект, содержащий узлы с историческими данными по скважинам\n
            well_name (str): имя скважины, которо принадлежат узелы \n

        Args:
            get_report (bool, optional): Определяет, нужно ли отображать отчет. Defaults to True.

        Returns:
            {
				"data_availability": bool (выполнялся или нет тест)
				"result": bool 
				"specification": {
						"result_mask" : np.array(int) shape(len(nodes), len(nodes[0])), 0 - поведение ряда BHP согласовано с поведением рядов LPR и GPR,\n
                                                       1 - поведение ряда BHP не согласовано с поведением ряда nodes[j] ,\n
                        "time_scale" : np.array(datetime.date), шкала времени для временного ряда
                        "test_name" : str, название теста
                        "error_decr": text, описание ошибки                                
                        "well_name": str, имя скважины
                        "node_names": list,  список имен узлов, над которыми проводился тест 
                        "segments" : list, список индексов точек изменения временного ряда
                        "poly1d_segments" : list, список апроксимирующих кривых, для каждого сегмента                                               
					}
			}
        """

        needed_nodes = self.types_of_nodes_for_test['test_imbalance_trends_with_BHP']
        nodes_well = self.nodes_obj.nodes_wells[well_name]

        if needed_nodes['base'] not in nodes_well:
            return {"data_availability": False}
        elif len(set(nodes_well.keys()) & set(needed_nodes['debit'] + needed_nodes['injection'])) < 1:
            return {"data_availability": False}

        idx_start = self.nodes_obj.init_idx_ts_wells[well_name]

        def fill_nodes(kind_nodes):
            nodes_list = list()
            for node in needed_nodes[kind_nodes]:
                if node in nodes_well:
                    nodes_list.append(nodes_well[node][idx_start:])
                    nodes_names.append(node)
            return nodes_list

        node_BHP = nodes_well[needed_nodes['base']][idx_start:]
        nodes_names = [needed_nodes['base']]
        nodes_debit = fill_nodes('debit')
        nodes_inj = fill_nodes('injection')
        n = len(node_BHP)
        num_nodes = len(nodes_names)
        result_mask = np.zeros((num_nodes, n))

        algo = rpt.Pelt(model="l2", min_size=3).fit(node_BHP)
        bic = 2 * np.log(n) * (0.25 * n) ** (0.5)
        if bic < 2:
            return {"data_availability": False}

        segments = algo.predict(bic)

        p_segments = list()
        x = np.arange(n)
        i_last = 0

        def check_imbalance(i, nodes_list, ofset, sign):
            for j in range(len(nodes_list)):
                z = np.polyfit(x[i_last:i], nodes_list[j][i_last:i], 1)
                p_j = np.poly1d(z)
                if sameSign(p[1], p_j[1]) == sign:
                    result_mask[0][i_last:i] = 1
                    result_mask[j + ofset][i_last:i] = 1
                ps.append(p_j)

        for i in segments:
            ps = list()
            z_BHP = np.polyfit(x[i_last:i], node_BHP[i_last:i], 1)
            p = np.poly1d(z_BHP)
            ps.append(p)
            check_imbalance(i, nodes_debit, ofset=1, sign=True)
            check_imbalance(i, nodes_inj, ofset=len(nodes_debit) + 1, sign=False)
            p_segments.append(ps)
            i_last = i

        first_zeros = np.zeros((num_nodes, idx_start))
        result_mask = np.concatenate([first_zeros, result_mask], axis=1)
        result = result_mask[0].sum() == 0

        if result:
            text = 'поведение ряда BHP согласовано с поведением рядов LPR и GPR'
            report_text = self.generate_report_text(text, 1)
        else:
            text = (
                'При тренде на снижении ВНР должен наблюдаться рост LPR/GPR (и наоборот).Отклонение от этого возможно, \n'
                'но оно должно проверяться на интерференцию, изменение подвижностей жидкостей, активность газовой шапки, \n'
                'но пока не в автоматическом режиме.')
            report_text = self.generate_report_text(text, 0)

        self.update_report(report_text)
        specification = self.get_specification(result_mask, "test_imbalance_trends_with_BHP", report_text, well_name,
                                               nodes_names)
        specification['segments'] = segments
        specification['poly1d_segments'] = p_segments

        if get_report:
            self.report_function['test_imbalance_trends_with_BHP'](specification)

        return {"data_availability": True, "result": result, "specification": specification}

    def test_imbalance_anomaly(self, well_name: str, get_report=True) -> dict:
        """
        Метод проверяет: При резком изменении ВНР должен быть отклик в противоположную сторону на дебите. \n

        Required data:
            self.nodes_obj (Nodes_historical_wells_data): обьект, содержащий узлы с историческими данными по скважинам\n
            well_name (str): имя скважины, которо принадлежат узелы \n

        Args:
            get_report (bool, optional): Определяет, нужно ли отображать отчет. Defaults to True.

        Returns:
            {
				"data_availability": bool (выполнялся или нет тест)
				"result": bool 
				"specification": {
						"result_mask" : np.array(int), 0 - поведение ряда BHP согласовано с поведением рядов LPR и GPR,\n
                                                       1 - поведение ряда BHP не согласовано с поведением ряда nodes[j],\n

                        "time_scale" : np.array(datetime.date), шкала времени для временного ряда
                        "test_name" : str, название теста
                        "error_decr": text, описание ошибки                                
                        "well_name": str, имя скважины
                        "node_names": list,  список имен узлов, над которыми проводился тест                                              
					}
			}
        """
        needed_nodes = self.types_of_nodes_for_test['test_imbalance_anomaly']
        nodes_well = self.nodes_obj.nodes_wells[well_name]

        if needed_nodes['base'] not in nodes_well:
            return {"data_availability": False}
        elif len(set(nodes_well.keys()) & set(needed_nodes['debit'] + needed_nodes['injection'])) < 1:
            return {"data_availability": False}

        def fill_nodes(kind_nodes):
            nodes_list = list()
            for node in needed_nodes[kind_nodes]:
                if node in nodes_well:
                    nodes_list.append(nodes_well[node])
                    nodes_names.append(node)
            return nodes_list

        node_BHP = nodes_well[needed_nodes['base']]
        nodes_names = [needed_nodes['base']]
        nodes_debit = fill_nodes('debit')
        nodes_inj = fill_nodes('injection')
        n = len(node_BHP)
        result_mask = np.zeros((len(nodes_names), n))

        anomalies_BHP = self.nodes_anomalies[well_name][nodes_names[0]]
        idx_anomalies_BHP = np.where(anomalies_BHP != 0)[0]
        print()

        def check_imbalance_anomaly(nodes_list, ofset, sign):
            for j in range(len(nodes_list)):
                anomalies_j = self.nodes_anomalies[well_name][nodes_names[j + ofset]]
                idx_anomalies = set(np.concatenate([idx_anomalies_BHP, np.where(anomalies_j != 0)[0]]))
                for idx in idx_anomalies:
                    if idx != 0:
                        diff_BHP = node_BHP[idx - 1] - node_BHP[idx]
                        diff_j = nodes_list[j][idx - 1] - nodes_list[j][idx]
                    else:
                        diff_BHP = node_BHP[idx + 1] - node_BHP[idx]
                        diff_j = nodes_list[j][idx + 1] - nodes_list[j][idx]

                    if sameSign(diff_BHP, diff_j) == sign:
                        result_mask[0][idx] = 1
                        result_mask[j + ofset][idx] = 1

        check_imbalance_anomaly(nodes_debit, ofset=1, sign=True)
        check_imbalance_anomaly(nodes_inj, ofset=len(nodes_debit) + 1, sign=False)
        result = result_mask[0].sum() == 0

        if result:
            text = 'поведение ряда BHP согласовано с поведением рядов дебета и закачки'
            report_text = self.generate_report_text(text, 1)
        else:
            text = ('При резком изменении ВНР должен быть отклик на дебите, \n'
                    '(резкое снижение ВНР влечет увеличение дебита (наоборот с закачкой) и обратная ситуация). \n')
            report_text = self.generate_report_text(text, 0)

        self.update_report(report_text)
        specification = self.get_specification(result_mask, "test_imbalance_anomaly", report_text, well_name,
                                               nodes_names)

        if get_report:
            self.report_function['test_imbalance_anomaly'](specification)
                          
        return {"data_availability": True, "result": result, "specification": specification}
    
    def get_report_first_order_test(self, specification: dict, saving: bool = True):
        """
        Визуализатор тестов первого порядка. \n

        Required data:
            specification (dict): спецификация полученная при выполнении теста\n
            
        Args:
            saving (bool): если True, то сохраняем рисунок в папке с отчетами.  Defaults to True.
  
        """
        
        well_name = specification['well_name']
        test_name = specification['test_name']
        node_name = specification['node_names'][0]
        

        print('\n'+specification['error_decr']+self.delimeter)

        fig = plt.figure()        
        if node_name != 'DATE':
            node = self.nodes_obj.nodes_wells[well_name][node_name]
            time_scale = self.nodes_obj.time_scale
            x = time_scale
            plt.xlabel('date')
        else: 
            node = self.nodes_obj.time_scale 
            x = np.arange(len(node))
            plt.ylabel('date')
                    
        plt.plot(x, node)
        idxs = np.where(specification['result_mask'] == 1)[0]
        for i in range(len(idxs)):
            plt.axvspan(x[idxs[i]], x[idxs[i]], color='red', alpha=1)

        plt.title(f'{well_name}: {node_name}\n {test_name}')
        plt.show() 
 
        if saving:
            file_name = f'{test_name}_{well_name}_{node_name}.png'
            path_out_file = os.path.join(self.folder_report,file_name)
            fig.savefig(path_out_file)
            plt.close()
                
    def get_report_test_LR_eq_sum_OR_and_WR(self, specification: dict, saving: bool = True):
        """
        Визуализатор для теста LPR = SUM(WPR +OPR). \n

        Required data:
            specification (dict): спецификация полученная при выполнении теста\n
        
        Args:
            saving (bool): если True, то сохраняем рисунок в папке с отчетами.  Defaults to True.

        """    
        well_name = specification['well_name']
        test_name = specification['test_name']
        nodes_names = specification['node_names']
        time_scale = self.nodes_obj.time_scale

        nodes = list()
        for name in nodes_names:
            nodes.append(self.nodes_obj.nodes_wells[well_name][name])

        print('\n'+specification['error_decr']+self.delimeter)

        fig, ax = plt.subplots()
        
        x = time_scale
        plt.xlabel('date')

        lines = []
        for i in range(len(nodes)):
            lines.append(ax.plot(x, nodes[i], label = nodes_names[i]))
        
        idxs = np.where(specification['result_mask'] == 1)[0]
        
        for i in range(len(idxs)):
            plt.axvspan(x[idxs[i]], x[idxs[i]], color='red', alpha=1)

        plt.title(f'{well_name}\n {test_name}')
        plt.legend()
        plt.show() 
 
        if saving:
            file_name = f'{test_name}_{well_name}.png'
            path_out_file = os.path.join(self.folder_report,file_name)
            fig.savefig(path_out_file)
            plt.close()  

    def get_report_test_imbalance_trends_with_BHP(self, specification: dict, saving: bool = True):
        """
        Визуализатор для теста несогласованное поведение рядов BHP, GPR, WPR. \n

        Required data:
            specification (dict): спецификация полученная при выполнении теста\n
        
        Args:
            saving (bool): если True, то сохраняем рисунок в папке с отчетами.  Defaults to True.

        """ 

        well_name = specification['well_name']
        test_name = specification['test_name']
        nodes_names = specification['node_names']
        segments = specification['segments']
        idx_start = self.nodes_obj.init_idx_ts_wells[well_name]
        time_scale = self.nodes_obj.time_scale[idx_start:]
        result_mask = specification['result_mask']
        ps = specification['poly1d_segments']

        def drow_poli1d(num_node, clr):
            ind_last = 0
            x = np.arange(len(time_scale))
            for i in range(len(segments)):
                ind_s = segments[i] 
                p = ps[i][num_node]
                line = plt.plot(time_scale[ind_last:ind_s],p(x[ind_last:ind_s]), '--', c = clr)
                ind_last = ind_s
            return line
            

        nodes = list()
        for name in nodes_names:
            nodes.append(self.nodes_obj.nodes_wells[well_name][name][idx_start:])

        print('\n'+specification['error_decr']+self.delimeter)

        node_BHP = nodes[0]
        num_nodes = len(nodes)
              
        for i in range(1, num_nodes):
            fig, ax_BHP = plt.subplots(figsize = (8,5), layout='constrained')
            plt.xlabel('date')
            ax_BHP.set_ylim(node_BHP.min()-1, node_BHP.max())
            plot_BHP = ax_BHP.plot(time_scale, node_BHP, color = 'red', label = nodes_names[0])
            line_BHP = drow_poli1d(0, 'r')    
         
            ax_i = ax_BHP.twinx()
            ax_i.set_ylim(nodes[i].min(), nodes[i].max())
            plot_i = ax_i.plot(time_scale, nodes[i], color = 'blue', label = nodes_names[i]) 
            line_i = drow_poli1d(i, 'blue')           
            ax_BHP.legend(handles = plot_BHP+plot_i+line_BHP+line_i, loc = 'best')

            plt.fill_between(time_scale, max(nodes[i].max(), node_BHP.max()), 0, where=result_mask[i][idx_start:] == 1, facecolor='red', interpolate=False, alpha=0.2)
            plt.title(f'{test_name}\n скважина: {well_name}\n несогласованность {nodes_names[0]} и {nodes_names[i]}')
            plt.show() 

            if saving:
                file_name = f'{test_name}_{well_name}_{nodes_names[0]}_{nodes_names[i]}.png'
                path_out_file = os.path.join(self.folder_report,file_name)
                fig.savefig(path_out_file)             
                plt.close()        

    def get_report_test_imbalance_anomaly(self, specification: dict,  saving: bool = True):
        """
        Визуализатор для теста несогласованное поведение рядов BHP, GPR, WPR. \n

        Required data:
            specification (dict): спецификация полученная при выполнении теста\n
        
        Args:
            saving (bool): если True, то сохраняем рисунок в папке с отчетами.  Defaults to True.

        """ 
        well_name = specification['well_name']
        test_name = specification['test_name']
        nodes_names = specification['node_names']
        result_mask = specification['result_mask']
        time_scale = self.nodes_obj.time_scale

        nodes = list()
        for name in nodes_names:
            nodes.append(self.nodes_obj.nodes_wells[well_name][name])

        print('\n'+specification['error_decr']+self.delimeter)
        
        node_BHP = nodes[0]
        num_nodes = len(nodes_names)
              
        for i in range(1, num_nodes):
            fig, ax_BHP = plt.subplots(figsize = (8,5), layout='constrained')
            plt.xlabel('date')
            ax_BHP.set_ylim(node_BHP.min(), node_BHP.max())
            plot_BHP = ax_BHP.plot(time_scale, node_BHP, color = 'green', label = nodes_names[0])

            idxs = np.where(result_mask[i] == 1)[0]
            for j in range(len(idxs)):
                plt.axvspan(time_scale[idxs[j]], time_scale[idxs[j]], color='red', alpha=1) 
                        
            ax_i = ax_BHP.twinx()
            ax_i.set_ylim(nodes[i].min(), nodes[i].max()+1)
            plot_i = ax_i.plot(time_scale, nodes[i], color = 'blue', label = nodes_names[i]) 
            ax_BHP.legend(handles = plot_BHP+plot_i, loc = 'best')
            
            plt.title(f'{test_name}\n скважина: {well_name}\n несогласованность {nodes_names[0]} и {nodes_names[i]}')
            plt.show() 

            if saving:
                file_name = f'{test_name}_{well_name}_{nodes_names[0]}_{nodes_names[i]}.png'
                path_out_file = os.path.join(self.folder_report,file_name)
                fig.savefig(path_out_file)             
                plt.close()                        