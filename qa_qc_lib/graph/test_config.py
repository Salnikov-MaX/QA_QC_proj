import json
from dataclasses import dataclass
from typing import List

from dacite import from_dict

from qa_qc_lib.graph.tools.data_map import KernFileMap, DataMap
from qa_qc_lib.graph.tools.graph import Graph


@dataclass
class TestInfo:
    test_key: str
    test_name_code: str
    ready_for_launch: bool


@dataclass
class KernTest:
    data_key: str
    tests: List[TestInfo]


@dataclass
class KernTestConfig:
    data: List[KernFileMap]
    tests: List[KernTest]


@dataclass
class MainTestConfig:
    kern: KernTestConfig


def read_main_test_config(main_test_path: str) -> MainTestConfig:
    with open(main_test_path, 'r', encoding='utf-8') as file:
        config_data = json.load(file)

    return from_dict(data_class=MainTestConfig, data=config_data)


def get_test(data_key: str, graph: Graph, data_keys: List[str]) -> List[TestInfo]:
    return [TestInfo(graph_node.test_key, graph_node.test_name_code, Graph.test_is_ready(graph_node, data_keys))
            for graph_node in graph.get_tests(data_key)]


def generate_launch_config(data_map: DataMap, graph: Graph):
    kern_data_keys = list()
    for kern_file in data_map.kern_files:
        kern_data_keys += [n.data_key for n in kern_file.map]
    kern_data_keys = list(set(kern_data_keys))

    kern_test = [KernTest(key, get_test(key, graph, kern_data_keys)) for key in kern_data_keys]
    kern = KernTestConfig(data_map.kern_files, kern_test)
    return MainTestConfig(kern)
