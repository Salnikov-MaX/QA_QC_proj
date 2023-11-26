import json
from typing import List

from qa_qc_lib.graph.test_config import MainTestConfig
from qa_qc_lib.graph.tools.graph import Graph
from qa_qc_lib.tests.kern_tests.data_preprocessing_kern import DataPreprocessing
from qa_qc_lib.tests.kern_tests.kern import QA_QC_kern


class LaunchTest:
    def __init__(self, graph: Graph, main_test_config: MainTestConfig):
        self.graph = graph
        self.main_test_config = main_test_config

        self.kern = self.init_kern(main_test_config)

    def init_kern(self, main_test_config: MainTestConfig) -> QA_QC_kern:
        data_pre = DataPreprocessing()
        data_file_path = 'results/test.xlsx'
        for kern_data in main_test_config.kern.data:
            columns_map = {m.data_key.split('|')[0]: f'{kern_data.data_path}->{m.data_column}' for m in kern_data.map}
            columns_map['MD'] = f'{kern_data.data_path}->{kern_data.md}'
            columns_map['Скважина'] = f'{kern_data.data_path}->{kern_data.well}'

            data_pre.process_data(columns_map)

        data_pre.save_to_excel(save_path=data_file_path)

        return QA_QC_kern('results', data_file_path=data_file_path)

    def start_tests(self, result_path):
        test_names: List[str] = []
        for tests in self.main_test_config.kern.tests:
            test_names += [t_info.test_name_code for t_info in tests.tests]

        report_data = self.kern.start_tests(test_names)
        with open(result_path, 'w+', encoding='utf-8') as file:
            json.dump(report_data, file, ensure_ascii=False)
