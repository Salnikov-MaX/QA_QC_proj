from typing import Optional, List

from qa_qc_lib.graph.data_map.SeismicMap import SeismicData
from qa_qc_lib.graph.graph import Graph
from qa_qc_lib.graph.test_config.SeismicTestConfig import SeismicTestConfig
from qa_qc_lib.graph.test_launcher.BaseLauncher import BaseLauncher
from qa_qc_lib.tests.seismic_tests.seismic import QA_QC_seismic


class SeismicLauncher(BaseLauncher):
    def __init__(self, seismic_config: SeismicTestConfig, seismic_data: SeismicData, graph: Graph):
        self.graph = graph
        self.config = seismic_config
        self.data = seismic_data

    def init_qa_qc(self, file_path: str):
        QA_QC_seismic()

    def start_qa_qc(self) -> List[dict]:
        all_reports: List[dict] = []
        for group in self.config.test_groups:
            for test in group.tests:
                graph_edge = self.graph.get_required_data_by_test_code_name(test.test_name_code)
                if 'test' not in graph_edge.test_name_code:
                    continue
                required_data_info = self.data.find_by_data_key(graph_edge.required_data[0])
                qa_qc = QA_QC_seismic(file_path=required_data_info.file_path)
                report = qa_qc.start_tests([test.test_name_code])
                all_reports += report

        return all_reports
