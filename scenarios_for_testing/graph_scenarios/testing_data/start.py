from qa_qc_lib.graph.test_launcher.launch_tests import LaunchTest
from qa_qc_lib.graph.test_config import read_main_test_config
from qa_qc_lib.graph.tools.graph import Graph

if __name__ == '__main__':
    test_config_path = '../generate_test_configs_from_available_data/data/result.json'
    graph_path = '../../../qa_qc_lib/graph/config/graph.json'

    graph = Graph(graph_path)
    main_test_config = read_main_test_config(test_config_path)

    launch_test = LaunchTest(graph, main_test_config)

    launch_test.start_tests('results/report.json')
