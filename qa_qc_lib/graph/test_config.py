import json
from dataclasses import dataclass
from typing import List, Optional

from dacite import from_dict

from qa_qc_lib.graph.tools.data_map import KernFileMap, DataMap, KernData
from qa_qc_lib.graph.tools.graph import Graph


@dataclass
class TestInfo:
    test_key: str
    test_name_code: str
    ready_for_launch: bool


@dataclass
class KernDataGroupTests:
    data_key: str
    tests: Optional[List[TestInfo]]


@dataclass
class KernTestConfig:
    tests: List[KernDataGroupTests]


@dataclass
class CubesTestConfig:
    tests: List[KernDataGroupTests]


@dataclass
class MainTestConfig:
    data: DataMap
    kern_config: KernTestConfig


def read_main_test_config(main_test_path: str) -> MainTestConfig:
    with open(main_test_path, 'r', encoding='utf-8') as file:
        config_data = json.load(file)

    return from_dict(data_class=MainTestConfig, data=config_data)


def get_test(data_key: str, graph: Graph, data_keys: List[str]) -> List[TestInfo]:
    return [TestInfo(graph_node.test_key, graph_node.test_name_code, Graph.test_is_ready(graph_node, data_keys))
            for graph_node in graph.get_tests(data_key)]


def generate_launch_config(data_map: DataMap, graph: Graph):
    kern = get_kern_section_config(data_map, graph)
    return MainTestConfig(data_map, kern)


def get_kern_section_config(data_map: DataMap, graph: Graph) -> KernTestConfig:
    kern_data_keys = list()
    for kern_file in data_map.kern.files:
        kern_data_keys += [n.data_key for n in kern_file.map]

    kern_data_keys = list(set(kern_data_keys))

    kern_test_groups = [KernDataGroupTests(key, get_test(key, graph, kern_data_keys)) for key in kern_data_keys]
    if data_map.settings.only_ready_for_launch_test:
        for kern_group in kern_test_groups:
            kern_group.tests = [kern_test for kern_test in kern_group.tests if kern_test.ready_for_launch]
    return KernTestConfig(kern_test_groups)
