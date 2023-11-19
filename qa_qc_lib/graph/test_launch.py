from abc import ABCMeta, abstractmethod
import json
import os.path
import time
from datetime import datetime

from nbformat import from_dict
from dataclasses import dataclass
from dacite import from_dict
from typing_extensions import List

from qa_qc_lib.graph.graph import EnumQAQCClass, GraphTest
from qa_qc_lib.graph.tools.read_map import DataInfo
from qa_qc_lib.tests.kern_tests.kern import QA_QC_kern
from qa_qc_lib.tests.seismic_tests.seismic import QA_QC_seismic


@dataclass
class TestConfig:
    test_name: str
    priority_data_for_launch: List[DataInfo]
    all_data_for_launch: List[List[DataInfo]]
    ready_for_launch: bool


class LaunchTest:
    def __init__(self, test_config: TestConfig, graph_test: GraphTest):
        self.test_config: TestConfig = test_config
        self.graph_test: GraphTest = graph_test

    def start_test(self, result_dir: str):
        switch = {
            EnumQAQCClass.Kern: LaunchTestKern,
            EnumQAQCClass.Seismic: QA_QC_seismic,
        }

        test_class = switch[self.graph_test.test_class_name]
        test_obj: LaunchTest = test_class(self.test_config, self.graph_test)

        if not os.path.isdir(result_dir):
            raise Exception(f'Директории: {result_dir} не существует!')

        test_method = getattr(test_obj, self.graph_test.test_method_name)

        result_data: dict = test_method()
        now_time = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        result_path = os.path.join(result_dir, f'{self.test_config.test_name}_{now_time}.json', 'w+')

        with open(result_path, 'w+') as file:
            json.dump(result_data, file)

        return result_data

    @staticmethod
    def read_configs(test_config_path: str) -> List[TestConfig]:
        with open(test_config_path, 'r', encoding='utf-8') as file:
            test_configs = json.load(file)

        tests = []
        for data_config in test_configs:
            for test_config in data_config['tests']:
                if test_config['ready_for_launch']:
                    tests.append(from_dict(data_class=TestConfig, data=test_config))

        return tests


class LaunchTestKern(LaunchTest):
    def test_open_porosity(self):
        self.config.priority_data_for_launch
        test_class = QA_QC_kern(
            # porosity_open=
        )
        test_class.test_porosity_open_vs_swl()

    def check_input(self):
        pass

    def test_monotony(self):
        pass

    def test_residual_water_saturation(self):
        pass

    def test_data_tampering(self):
        pass

    def test_sg(self):
        pass

    def test_sgl(self):
        pass

    def test_so(self):
        pass

    def test_sogcr(self):
        pass

    def test_parallel_permeability(self):
        pass


class LaunchSeismic:
    def __init__(self, test_config_path: str):
        pass
