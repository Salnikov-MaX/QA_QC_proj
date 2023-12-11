from typing import Optional

import numpy as np
import os

from matplotlib import pyplot as plt
from qa_qc_lib.tests.base_test import QA_QC_main
from qa_qc_lib.tests.gis.gis_nodes import Nodes_gis_data


class QA_QC_gis(QA_QC_main):
    def __init__(self, nodes_obj: Nodes_gis_data, folder_report: str):
        """
        Класс с тестами первого и второго порядка над историческими данными по скважинам

        Args:
            nodes_obj (Nodes_gis_data): обьект, содержащий узлы с данными ГИС\РЕГИС по скважинам. \n
            folder_report (str): путь до папки с отчетамию \n
        """
        super().__init__()
        self.folder_report = folder_report
        self.nodes_obj = nodes_obj
        self.types_of_nodes_for_test = {'test_limit_0_47': ['Кп_абс', 'Кп_октр', 'Кп_эфф'],
                                        'test_missing_intervals': list(nodes_obj.reader.mnemonics.keys()),
                                        'test_overlap_intervals': list(nodes_obj.reader.mnemonics.keys())}

        self.order_tests = {1: ['test_limit_0_47'],
                            2: ['test_missing_intervals', 'test_overlap_intervals']}

        self.report_function = {'test_limit_0_47': self.get_report_tests,
                                'test_missing_intervals': self.get_report_tests,
                                'test_overlap_intervals': self.get_report_tests}

    def get_specification(self, result_mask: np.array, test_name: str, error_decr: str, node_name: str,
                          logs_names: list):

        """Возвращает спецификацию теста

        Required data:
            result_mask (np.array(int)): результат работы теста,\n
            test_name (str):  название теста
            error_decr (str): описание ошибки   
            node_name (str): имя узла 
            logs_names (list): список каротажей, над которыми проводился тест   

        Returns:
            specification: {
                        "result_mask" : np.array(int), 0 - значение правильное,\n
                                                       1 - значение ошибочное,\n
                        "depth" : np.array(float), шкала глубин каротажа
                        "test_name" : str, название теста
                        "error_decr": text, описание ошибки   
                        "well_name": str, имя скважины
                        "node_name": str, имя узла
                        "node_names": list, список каротажей                                               
                    }
            }
        """

        return {"result_mask": result_mask,
                "depth": self.nodes_obj.dept,
                "test_name": test_name,
                "error_decr": error_decr,
                "well_name": self.nodes_obj.well_name,
                "node_name": node_name,
                "logs_names": logs_names}

    def test_limit_0_47(self, node_name: str, get_report=True) -> dict:
        """
        Метод проверяет каротажи пористости на соответствие интервалу (0 ; 47.6]

        Required data:
            node_name (str): имя типа узла \n
            
        Args:
            get_report (bool, optional): Определяет, нужно ли отображать отчет. Defaults to True.

        Returns:
			{
				"data_availability": bool (выполнялся или нет тест)
				"result": bool 
				"specification": {
						"result_mask" : np.array(int), 0 - значение временного ряда лежит в пределах интервала (0 ; 47.6],\n
                                                       1 - значение временного ряда находится за пределами интервала (0 ; 47.6],\n
                        "depth" : np.array(float), шкала глубин каротажа
                        "test_name" : str, название теста
                        "error_decr": text, описание ошибки   
                        "well_name": str, имя скважины
                        "node_name": str, имя узла
                        "logs_names": list, список каротажей                                                 
					}
			}
        """

        if not node_name in self.types_of_nodes_for_test['test_limit_0_47']:
            return {"data_availability": False,
                    'specification': {
                        "node_name": node_name,
                        'test_name': 'test_limit_0_47',
                        "well_name": self.nodes_obj.well_name,
                    },
                    }

        logs = self.nodes_obj.gis_nodes[node_name]
        result_mask = []
        logs_names = []
        for name_log, val in logs.items():
            mask_0 = val <= 0
            mask_1 = val > 47.6
            result_mask.append((mask_0 + mask_1).astype(int))
            logs_names.append(name_log)

        result_mask = np.array(result_mask)
        result = result_mask.sum() == 0

        if result:
            text = 'Значения ряда находяться в интервале (0 ; 47.6]'
            report_text = self.generate_report_text(text, 1)
        else:
            text = 'Некоторые значения ряда находяться за пределами интервала (0 ; 47.6]'
            report_text = self.generate_report_text(text, 0)

        self.update_report(report_text)
        specification = self.get_specification(result_mask, "test_limit_0_47", report_text, node_name, logs_names)

        if get_report:
            self.report_function['test_limit_0_47'](report_text, specification)

        return {"data_availability": True, "result": bool(result), "specification": specification}

    def test_missing_intervals(self, node_name: str, get_report=True) -> dict:
        """
        Метод проверяет пропуск записи ГИС в интервале пласта \n

        Required data:
            node_name (str): имя типа узла \n
            
        Args:
            get_report (bool, optional): Определяет, нужно ли отображать отчет. Defaults to True.

        Returns:
			{
				"data_availability": bool (выполнялся или нет тест)
				"result": bool 
				"specification": {
						"result_mask" : np.array(int), 0 - значение заполнено,\n
                                                       1 - пропуск,\n
                        "depth" : np.array(float), шкала глубин каротажа
                        "test_name" : str, название теста
                        "error_decr": text, описание ошибки   
                        "well_name": str, имя скважины
                        "node_name": str, имя узла
                        "logs_names": list, список каротажей                                                 
					}
			}
        """

        if not node_name in self.types_of_nodes_for_test['test_missing_intervals']:
            return {"data_availability": False,
                    'specification': {
                        "node_name": node_name,
                        'test_name': 'test_missing_intervals',
                        "well_name": self.nodes_obj.well_name,
                    },
                    }

        logs = self.nodes_obj.gis_nodes[node_name]
        result_mask = []
        logs_names = []
        for name_log, val in logs.items():
            result_mask.append(np.isnan(val).astype(int))
            logs_names.append(name_log)

        result_mask = np.array(result_mask)
        result = result_mask.sum() == 0

        if result:
            text = 'В каротажах отстутсвуют пропуски записи ГИС в интервале пласта '
            report_text = self.generate_report_text(text, 1)
        else:
            text = 'В некоторых каротажах есть пропуски записи ГИС в интервале пласта '
            report_text = self.generate_report_text(text, 0)

        self.update_report(report_text)
        specification = self.get_specification(result_mask, "test_missing_intervals", report_text, node_name,
                                               logs_names)

        if get_report:
            self.report_function['test_missing_intervals'](report_text, specification)

        return {"data_availability": True, "result": bool(result), "specification": specification}

    def test_overlap_intervals(self, node_name: str, get_report=True) -> dict:
        """
        Метод проверяет перекрытие интервалов записи для основной и повторной записи ГИС  \n

        Required data:
            node_name (str): имя типа узла \n
            
        Args:
            get_report (bool, optional): Определяет, нужно ли отображать отчет. Defaults to True.

        Returns:
			{
				"data_availability": bool (выполнялся или нет тест)
				"result": bool 
				"specification": {
						"result_mask" : np.array(int), 0 - нет перекрытия,\n
                                                       1 - есть перекрытие,\n
                        "depth" : np.array(float), шкала глубин каротажа
                        "test_name" : str, название теста
                        "error_decr": text, описание ошибки   
                        "well_name": str, имя скважины
                        "node_name": str, имя узла
                        "logs_names": list, список каротажей                                                 
					}
			}
        """

        if not node_name in self.types_of_nodes_for_test['test_overlap_intervals']:
            return {"data_availability": False,
                    'specification': {
                        "node_name": node_name,
                        'test_name': 'test_overlap_intervals',
                        "well_name": self.nodes_obj.well_name,
                    },
                    }

        logs = self.nodes_obj.gis_nodes[node_name]
        logs_names = list(logs.keys())

        result_mask = (np.isnan(np.array(list(logs.values()))).astype(int) - 1) * (-1)
        result_mask = np.sum(result_mask, axis=0) > 1
        result_mask = result_mask.astype(int)
        result = result_mask.sum() == 0

        if result:
            text = 'Перекрытия интервалос записей ГИС не обнаружено.'
            report_text = self.generate_report_text(text, 1)
        else:
            text = 'Обнаружены перекрытия интервалов записей ГИС'
            report_text = self.generate_report_text(text, 0)

        self.update_report(report_text)
        specification = self.get_specification(result_mask, "test_overlap_intervals", report_text, node_name,
                                               logs_names)

        if get_report:
            self.report_function['test_overlap_intervals'](report_text, specification)

        return {"data_availability": True, "result": bool(result), "specification": specification}

    def get_report_tests(self, specification: dict, saving: bool = True) -> Optional[str]:
        """
        Визуализатор для всех тестов. \n

        Required data:
            specification (dict): спецификация полученная при выполнении теста\n
            
        Args:
            saving (bool): если True, то сохраняем рисунок в папке с отчетами.  Defaults to True.
  
        """
        well_name = self.nodes_obj.well_name
        depth = specification['depth']
        test_name = specification['test_name']
        node_name = specification['node_name']
        logs_names = specification['logs_names']
        result_mask = specification['result_mask']
        num_logs = len(logs_names)

        print('\n' + specification['error_decr'] + self.delimeter)

        fig, ax = plt.subplots(1, num_logs)
        fig.tight_layout(w_pad=3)
        fig.suptitle(f'тест: {test_name}\n узел: {node_name}\n', y=1.1)
        if num_logs == 1:
            ax = [ax]

        for i in range(num_logs):
            log = self.nodes_obj.gis_nodes[node_name][logs_names[i]]
            ax[i].plot(log, depth)

            if test_name == 'test_limit_0_47':
                ax[i].axvspan(47.6, 47.6, color='red', alpha=1)
                min_x = 47.6
            else:
                min_x = 0

            if len(result_mask.shape) == 1:
                error_mask = result_mask
            else:
                error_mask = result_mask[i]

            ax[i].fill_betweenx(depth, np.nan_to_num(log).max(), min_x, where=error_mask == 1, facecolor='red',
                                interpolate=False, alpha=0.2)
            ax[i].set_title('каротаж: ' + logs_names[i], fontsize=10)
            ax[i].set_ylabel('depth')
            ax[i].invert_xaxis()
            ax[i].invert_yaxis()
            ax[i].set_ylim(depth.max(), depth.min())

        if saving:
            file_name = f'{test_name}_{well_name}_{node_name}.png'
            path_out_file = os.path.join(self.folder_report, file_name)
            fig.savefig(path_out_file)
            return path_out_file
        else:
            plt.show()
