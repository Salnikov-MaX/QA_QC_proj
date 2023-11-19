from typing import List

from qa_qc_lib.graph.graph import GraphTest
from qa_qc_lib.graph.test_launch import LaunchTest, TestConfig

if __name__ == '__main__':
    # launch_test = LaunchTest()
    test_config_path = 'data/result.json'
    graph_path = '../../../qa_qc_lib/graph/config/graph.json'
    test_configs: List[TestConfig] = LaunchTest.read_configs(test_config_path)

    graph_dict: dict[str, GraphTest] = GraphTest.read_tests_info_file_as_dict(graph_path)
    launch_tests: List[LaunchTest] = [LaunchTest(config, graph_dict[config.test_name]) for config in test_configs]

    for test in launch_tests:
        print(test)

    exit(0)
