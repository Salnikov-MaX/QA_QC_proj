import json
import os.path
import shutil
import sys
import time
from typing import List, Optional

import numpy as np

from qa_qc_lib.graph.graph import Graph
from qa_qc_lib.graph.test_config import MainTestConfig
from qa_qc_lib.graph.test_launcher.BaseLauncher import BaseLauncher
from qa_qc_lib.graph.test_launcher.CubesLauncher import CubeLauncher
from qa_qc_lib.graph.test_launcher.GisLauncher import GisLauncher
from qa_qc_lib.graph.test_launcher.KernLauncher import KernLauncher
from qa_qc_lib.graph.test_launcher.SeismicLauncher import SeismicLauncher
from qa_qc_lib.graph.test_launcher.WellLauncher import WellLauncher


class LaunchTest:
    def __init__(self, main_test_config: MainTestConfig, graph: Optional[Graph] = None):
        self.graph = graph or Graph()
        self.main_test_config = main_test_config

    @staticmethod
    def get_launchers(graph: Graph, config: MainTestConfig, temp_dir: str) -> List[BaseLauncher]:
        launchers: List[BaseLauncher] = []
        if config.data.kern and config.kern_config:
            launchers.append(KernLauncher(graph, config.kern_config, config.data.kern))

        if config.data.cube and config.cubes_config:
            launchers.append(CubeLauncher(graph, config.cubes_config, config.data.cube))

        if config.data.well and config.well_config:
            launchers.append(WellLauncher(config.well_config, config.data.well, temp_dir))

        if config.data.seismic is not None or config.seismic_config:
            launchers.append(SeismicLauncher(config.seismic_config, config.data.seismic, graph))

        if config.data.gis is not None or config.gis_config:
            launchers.append(GisLauncher(config.gis_config, config.data.gis, temp_dir))

        return launchers

    @staticmethod
    def filter_report(report: dict) -> list[str]:
        pass

    @staticmethod
    def move_report_data_to_report_dir(data_reports_dir: str, report: dict):
        if report.get('report_data'):
            base_name = f'{report["report_id"]}_' + os.path.basename(report['report_data'])
            new_place = os.path.join(data_reports_dir, base_name)

            os.replace(report['report_data'], new_place)
            report['report_data'] = new_place

    @staticmethod
    def prepare_data_for_json_convert(report: dict):
        if not report.get('specification'):
            return

        for k in report['specification'].keys():
            if isinstance(report['specification'][k], np.ndarray):
                report['specification'][k] = report['specification'][k].tolist()
            try:
                json.dumps({k: report['specification'][k]})
            except:
                report['specification'][k] = str(report['specification'][k])

    @staticmethod
    def putting_big_data_in_a_separate_file(detailed_reports_dir: str, report: dict) -> Optional[str]:
        spec: dict = report.get('specification')

        if spec is None:
            return

        keys_for_pop = [s_key for s_key in spec.keys() if not isinstance(spec[s_key], str)]
        keys_for_pop = [s_key for s_key in keys_for_pop if sys.getsizeof(spec[s_key]) > 50]

        if keys_for_pop:
            spec_file_path = os.path.join(detailed_reports_dir, f'{report["report_id"]}.json')
            with open(spec_file_path, 'w+', encoding='utf-8') as file:
                json.dump(report, file, ensure_ascii=False)

            [report['specification'].pop(s_key) for s_key in keys_for_pop]
            return spec_file_path

    @staticmethod
    def make_report(result_dir: str, report_data: List[dict], report_name: str = 'reports'):
        time_string = time.strftime("%d_%m_%Y_%H_%M_%S", time.localtime())

        report_dir = os.path.join(result_dir, f'{report_name}_{time_string}')
        data_reports_dir = os.path.join(report_dir, 'report_data')
        detailed_reports_dir = os.path.join(report_dir, 'detailed_reports')

        if os.path.isdir(report_dir):
            shutil.rmtree(report_dir)

        for dir_path in [report_dir, data_reports_dir, detailed_reports_dir]:
            os.mkdir(dir_path)

        for report_id, report in enumerate(report_data):
            report['report_id'] = report_id

            LaunchTest.move_report_data_to_report_dir(data_reports_dir, report)
            LaunchTest.prepare_data_for_json_convert(report)

            report['specification_file'] = LaunchTest.putting_big_data_in_a_separate_file(detailed_reports_dir, report)

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
