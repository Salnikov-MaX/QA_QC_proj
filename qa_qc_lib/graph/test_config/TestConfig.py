from dataclasses import dataclass
from typing import Optional, List

from qa_qc_lib.graph.data_map.DataMap import DataMap
from qa_qc_lib.graph.graph import Graph


@dataclass
class TestInfo:
    test_key: str
    test_name_code: str
    ready_for_launch: bool


@dataclass
class DataGroupTests:
    data_key: str
    tests: Optional[List[TestInfo]]


def get_test(data_key: str, graph: Graph, data_keys: List[str]) -> List[TestInfo]:
    return [TestInfo(graph_node.test_key, graph_node.test_name_code, Graph.test_is_ready(graph_node, data_keys))
            for graph_node in graph.get_tests(data_key)]


def get_data_groups(data_map: DataMap, graph: Graph, data_keys: [str]) -> List[DataGroupTests]:
    kern_test_groups = [DataGroupTests(key, get_test(key, graph, data_keys)) for key in data_keys]
    if data_map.settings.only_ready_for_launch_test:
        for kern_group in kern_test_groups:
            kern_group.tests = [kern_test for kern_test in kern_group.tests if kern_test.ready_for_launch]

    return kern_test_groups
