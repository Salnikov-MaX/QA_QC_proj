from typing import Optional, List

from qa_qc_lib.graph.test_config import KernTestConfig
from qa_qc_lib.graph.test_launcher.BaseLauncher import BaseLauncher
from qa_qc_lib.graph.data_map.DataMap import KernData
from qa_qc_lib.graph.data_map.KernMap import KernPathInfo, DataFilter
from qa_qc_lib.graph.graph import Graph
from qa_qc_lib.tests.kern_tests.data_kern import DataKern
from qa_qc_lib.tests.kern_tests.data_preprocessing_kern import DataPreprocessing
from qa_qc_lib.tests.kern_tests.kern import QA_QC_kern


class KernLauncher(BaseLauncher):
    def __init__(self, graph: Graph, kern_config: KernTestConfig, kern_data: KernData):
        self.graph = graph
        self.kern_config = kern_config
        self.kern_data = kern_data

    @staticmethod
    def kern_data_path_generate(file_path: str, kern_data: KernPathInfo):
        path_items = [file_path, kern_data.sheet_name, kern_data.data_column]
        return "->".join([p_i for p_i in path_items if p_i is not None])

    @staticmethod
    def init_kern(kern_data: KernData) -> QA_QC_kern:
        data_pre = DataPreprocessing()
        data_file_path = 'data/data_file.xlsx'
        for kern_data in kern_data.files:
            columns_map = {m.data_key.split('|')[0]: KernLauncher.kern_data_path_generate(kern_data.data_path, m) for m
                           in kern_data.map}

            data_pre.process_data(columns_map, data_file_path)

        return QA_QC_kern('data', data_file_path=data_file_path)

    @staticmethod
    def get_filters(groups: List[str], qa_qc_data_kern: DataKern) -> List[List[DataFilter]]:
        filters: List[List[DataFilter]] = []
        groups = [g.split('|')[0] for g in groups]

        for row in qa_qc_data_kern.get_attributes(groups).drop_duplicates(subset=groups).to_dict(orient='records'):
            filter_group: List[DataFilter] = []
            for key in row.keys():
                filter_group.append(DataFilter(key, row[key], '=='))

            filters.append(filter_group)

        return filters

    def start_qa_qc(self) -> [Optional[dict]]:
        kern_qa_qc = self.init_kern(self.kern_data)

        test_names: List[str] = []
        for group in self.kern_config.test_groups:
            test_names += [t_info.test_name_code for t_info in group.tests if t_info.ready_for_launch]

        if not self.kern_data.group_by_columns:
            kern_report_data = kern_qa_qc.start_tests(test_names)

        else:
            filters = KernLauncher.get_filters(self.kern_data.group_by_columns, kern_qa_qc.data_kern)
            tests_with_filters: List[tuple[str, List[dict]]] = []
            for test_name in test_names:
                for group_filter in filters:
                    test_filters: List[dict] = [dict({"name": g.column_name,
                                                      "value": g.value,
                                                      "operation": g.operation})
                                                for g in group_filter]

                    tests_with_filters.append((test_name, test_filters))
            kern_report_data = kern_qa_qc.start_tests_with_filters(tests_with_filters, get_report=False)

        return kern_report_data
