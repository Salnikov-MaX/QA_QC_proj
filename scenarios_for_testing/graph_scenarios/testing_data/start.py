import os
from typing import List

from qa_qc_lib.graph.graph import GraphTest
from qa_qc_lib.graph.test_launch import LaunchTest, TestConfig

if __name__ == '__main__':
    # launch_test = LaunchTest()
    test_config_path = '../generate_test_configs_from_available_data/data/result.json'
    graph_path = '../../../qa_qc_lib/graph/config/graph.json'

    graph_dict: dict[str, GraphTest] = GraphTest.read_tests_info_file_as_dict(graph_path)
    test_configs: List[TestConfig] = LaunchTest.read_configs(test_config_path, list(graph_dict.values()))

    launch_tests: List[LaunchTest] = [LaunchTest(config) for config in test_configs]

    results_dir = 'results'
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    for test in launch_tests:
        print(test.start_test(results_dir))

    exit(0)
