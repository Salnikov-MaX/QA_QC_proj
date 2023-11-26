import json
from typing import List, Optional

from qa_qc_lib.graph.test_config import MainTestConfig
from qa_qc_lib.graph.test_launcher.BaseLauncher import BaseLauncher
from qa_qc_lib.graph.test_launcher.KernLauncher import KernLauncher
from qa_qc_lib.graph.tools.graph import Graph


class LaunchTest:
    def __init__(self, graph: Graph, main_test_config: MainTestConfig):
        self.graph = graph
        self.main_test_config = main_test_config
        self.launchers: List[BaseLauncher] = [
            KernLauncher(graph, main_test_config.kern_config, main_test_config.data.kern)
        ]

    def start_tests(self, result_path):
        all_reports: [Optional[dict]] = []

        for launcher in self.launchers:
            launcher_report = launcher.start_qa_qc()
            all_reports += launcher_report
        # kern_reports = self.start_kern_section()
        # cubes_reports = self.start_cubes_section()

        with open(result_path, 'w+', encoding='utf-8') as file:
            json.dump(all_reports, file, ensure_ascii=False)
