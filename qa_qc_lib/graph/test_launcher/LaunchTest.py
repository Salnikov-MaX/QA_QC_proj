import json
import os.path
import shutil
import sys
import time
from datetime import datetime
from typing import List, Optional

from qa_qc_lib.graph.test_config import MainTestConfig
from qa_qc_lib.graph.test_launcher.BaseLauncher import BaseLauncher
from qa_qc_lib.graph.test_launcher.CubesLauncher import CubeLauncher
from qa_qc_lib.graph.test_launcher.KernLauncher import KernLauncher
from qa_qc_lib.graph.graph import Graph
from qa_qc_lib.graph.test_launcher.WellLauncher import WellLauncher


class LaunchTest:
    def __init__(self, main_test_config: MainTestConfig, graph: Optional[Graph] = None):
        self.graph = graph or Graph()
        self.main_test_config = main_test_config

    @staticmethod
    def get_launchers(graph: Graph, main_test_config: MainTestConfig, temp_dir: str) -> List[BaseLauncher]:
        launchers: List[BaseLauncher] = []
        if main_test_config.data.kern is not None:
            launchers.append(KernLauncher(graph, main_test_config.kern_config, main_test_config.data.kern))

        if main_test_config.data.cube is not None:
            launchers.append(CubeLauncher(graph, main_test_config.cubes_config, main_test_config.data.cube))

        if main_test_config.data.well is not None:
            launchers.append(WellLauncher(main_test_config.well_config, main_test_config.data.well, temp_dir))

        return launchers

    @staticmethod
    def make_report(result_dir: str, report_data: List[dict], report_name: str = 'reports'):
        time_string = time.strftime("%d_%m_%Y_%H_%M_%S", time.localtime())

        report_dir = os.path.join(result_dir, f'{report_name}_{time_string}')
        if os.path.isdir(report_dir):
            shutil.rmtree(report_dir)

        os.mkdir(report_dir)

        detailed_reports_dir = os.path.join(report_dir, 'detailed_reports')
        os.mkdir(detailed_reports_dir)

        data_reports_dir = os.path.join(report_dir, 'report_data')
        os.mkdir(data_reports_dir)

        for report_id, report in enumerate(report_data):
            report['report_id'] = report_id

            if report.get('report_data'):
                base_name = f'{report_id}_' + os.path.basename(report['report_data'])
                new_place = os.path.join(data_reports_dir, base_name)

                os.replace(report['report_data'], new_place)

                report['report_data'] = new_place

            keys_for_pop = None
            if report.get('specification'):
                keys_for_pop = [s_key for s_key in report.get('specification') if
                                sys.getsizeof(report['specification'][s_key]) > 50]

            if keys_for_pop:
                spec_file = os.path.join(detailed_reports_dir, f'{report_id}.json')
                with open(spec_file, 'w+', encoding='utf-8') as file:
                    json.dump(report, file, ensure_ascii=False)

                report['specification_file'] = spec_file
                [report['specification'].pop(s_key) for s_key in keys_for_pop]

        with open(os.path.join(report_dir, 'main_report.json'), 'w+', encoding='utf-8') as file:
            json.dump(report_data, file, ensure_ascii=False)

    def start_tests(self, result_dir: str):
        temp_dir = 'temp'
        if not os.path.isdir(result_dir):
            os.mkdir(result_dir)

        launchers: List[BaseLauncher] = self.get_launchers(self.graph, self.main_test_config, temp_dir)

        for launcher in launchers:
            launcher_report = launcher.start_qa_qc()
            report_name = f'report_{launcher.__class__.__name__.replace("Launcher", "")}'
            self.make_report(result_dir, launcher_report, report_name=report_name)

        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)
