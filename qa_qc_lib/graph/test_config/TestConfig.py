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
    def get_test_info(graph_node):
        return TestInfo(graph_node.test_key,
                        graph_node.test_name_code,
                        Graph.test_is_ready(graph_node, data_keys))

    return [get_test_info(graph_node) for graph_node in graph.get_tests(data_key)]


def get_data_groups(data_map: DataMap, graph: Graph, data_keys: [str]) -> List[DataGroupTests]:
    test_groups = [DataGroupTests(key, get_test(key, graph, data_keys)) for key in data_keys]
    if data_map.settings.only_ready_for_launch_test:
        for group in test_groups:
            group.tests = [test for test in group.tests if test.ready_for_launch]

    return test_groups
