from qa_qc_lib.graph.graph import GraphTest
from qa_qc_lib.graph.test_launch import LaunchTest

if __name__ == '__main__':
    launch_test = LaunchTest('data/result.json')

    graph = GraphTest.read_tests_info_file_as_dict('../../../qa_qc_lib/graph/config/graph.json')
    x = 1