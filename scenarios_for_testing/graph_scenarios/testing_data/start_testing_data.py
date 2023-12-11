import json

from dacite import from_dict

from qa_qc_lib.graph.test_launcher.LaunchTest import LaunchTest
from qa_qc_lib.graph.test_config.MainTestConfig import MainTestConfig
from qa_qc_lib.graph.graph import Graph

if __name__ == '__main__':
    test_config_path = '../generate_test_configs/data/result.json'
    graph_path = '../generate_test_configs/data/edges.json'

    graph = Graph(graph_path)
    with open(test_config_path, 'r', encoding='utf-8') as file:
        main_test_config = from_dict(data_class=MainTestConfig, data=json.load(file))

    launch_test = LaunchTest(main_test_config, graph)

    launch_test.start_tests('results')
