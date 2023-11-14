import json

from qa_qc_lib.graph.graph import GraphTest
from qa_qc_lib.graph.tools.read_map import DataMap
from scenarios_for_testing.graph_scenarios import project_path

if __name__ == '__main__':
    map_path = 'data/map.json'
    data_keys_path = f'{project_path}\\qa_qc_lib\\graph\\config\\data_keys.txt'
    graph_path = f'{project_path}\\qa_qc_lib\\graph\\config\\graph.json'

    data_map = DataMap(map_path, data_keys_path)
    graph_tests = GraphTest.read_tests_info_file(graph_path)

    data_test_dict = {
        d.file_key: GraphTest.get_tests_for_data(d.file_key, graph_tests)
        for d in data_map.files_info
    }

    test_configs = []
    for key in data_test_dict.keys():
        tests_configs = [d.get_test_config(data_map.files_info) for d in data_test_dict[key]]

        if not data_map.settings.show_tests_not_ready_for_launch:
            tests_configs = [t_c for t_c in tests_configs if t_c['ready_for_launch']]

        test_configs.append({
            "file_key": key,
            "tests": [test_config for test_config in tests_configs]
        })

    with open('data/result.json', 'w+', encoding='utf-8') as file:
        json.dump(test_configs, file, ensure_ascii=False)
