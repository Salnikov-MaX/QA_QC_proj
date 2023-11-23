from abc import ABCMeta, abstractmethod
import json
import os.path
import time
from datetime import datetime

from dataclasses import dataclass
from json import JSONEncoder

import numpy
from dacite import from_dict
from typing import List

from qa_qc_lib.graph.graph import EnumQAQCClass, GraphTest
from qa_qc_lib.graph.tools.read_map import DataInfo
from qa_qc_lib.tests.kern_tests.kern import QA_QC_kern
from qa_qc_lib.tests.seismic_tests.seismic import QA_QC_seismic


@dataclass
class TestConfig:
    priority_data_for_launch: List[DataInfo]
    all_data_for_launch: List[List[DataInfo]]
    ready_for_launch: bool
    graph_test: GraphTest


class LaunchTest:
    def __init__(self, test_config: TestConfig):
        self.test_config: TestConfig = test_config

    def start_test(self, result_dir: str):
        switch_class = {
            EnumQAQCClass.Kern: LaunchTestKern,
            EnumQAQCClass.Seismic: LaunchSeismic,
        }

        test_class = switch_class[self.test_config.graph_test.test_class_name]
        test_obj: LaunchTest = test_class(self.test_config)

        if not os.path.isdir(result_dir):
            raise Exception(f'Директории: {result_dir} не существует!')

        test_method = getattr(test_obj, self.test_config.graph_test.test_method_name)

        result_data: dict = test_method()
        result_data = result_data if result_data is not None else {
            "test_method": self.test_config.graph_test.test_method_name}

        now_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        test_name = self.test_config.graph_test.test_method_name
        result_path = os.path.join(result_dir, f'{test_name}_{now_time}.json')

        with open(result_path, 'w+', encoding='utf-8') as file:
            json.dump(result_data, file, ensure_ascii=False,
                      default=lambda o: f"<<non-serializable: {type(o).__qualname__}>>")

        return test_name, self.test_config

    @staticmethod
    def read_configs(test_config_path: str, graph_list: List[GraphTest]) -> List[TestConfig]:
        with open(test_config_path, 'r', encoding='utf-8') as file:
            test_configs = json.load(file)

        graph_dict = {node.test_code_name: node for node in graph_list}

        tests = []
        for data_config in test_configs:
            for test_config in data_config['tests']:
                if test_config['ready_for_launch']:
                    tests.append(from_dict(data_class=TestConfig,
                                           data={**test_config,
                                                 "graph_test": graph_dict[test_config['test_name']]}))

        return tests


class LaunchTestKern(LaunchTest):
    def test_open_porosity(self) -> dict:
        data_info: [DataInfo] = self.test_config.priority_data_for_launch
        test_class = QA_QC_kern(
            file_path='result',
            porosity_open=data_info[0]
        )
        return test_class.test_porosity_open_vs_swl()

    def check_input(self) -> dict:
        pass

    def test_monotony(self) -> dict:
        data_info: [DataInfo] = self.test_config.priority_data_for_launch
        test_class = QA_QC_kern(
            file_path='result',
            depth=data_info[0]
        )
        return test_class.test_monotony()

    def test_residual_water_saturation(self) -> dict:
        pass

    def test_data_tampering(self) -> dict:
        pass

    def test_sg(self) -> dict:
        pass

    def test_sgl(self) -> dict:
        pass

    def test_so(self) -> dict:
        pass

    def test_sogcr(self) -> dict:
        pass

    def test_parallel_permeability(self) -> dict:
        pass


class LaunchSeismic:
    def __init__(self, test_config_path: str):
        pass
