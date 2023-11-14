import json

from nbformat import from_dict
from dataclasses import dataclass
from dacite import from_dict
from typing_extensions import List


@dataclass
class DataInfo:
    file_key: str
    file_path: str


@dataclass
class TestConfig:
    test_name: str
    priority_data_for_launch: List[DataInfo]
    all_data_for_launch: List[List[DataInfo]]
    ready_for_launch: bool


class LaunchTest:
    def __init__(self, test_config_path: str):
        self.config = LaunchTest.read_configs(test_config_path)

    @staticmethod
    def read_configs(test_config_path: str) -> [TestConfig]:
        with open(test_config_path, 'r', encoding='utf-8') as file:
            test_configs = json.load(file)

        tests = []
        for data_config in test_configs:
            for test_config in data_config['tests']:
                if test_config['ready_for_launch']:
                    tests.append(from_dict(data_class=TestConfig, data=test_config))

        return tests
        # return [t for t in tests if t.ready_for_launch]
