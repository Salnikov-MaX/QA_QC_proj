import json

from qa_qc_lib.graph.graph import GraphTest
from qa_qc_lib.graph.tools.read_map import DataMap
from scenarios_for_testing.graph_scenarios import project_path

if __name__ == '__main__':
    map_path = 'data/map.json'
    data_keys_path = f'{project_path}qa_qc_lib\\graph\\config\\data_keys.txt'
    graph_path = f'{project_path}qa_qc_lib\\graph\\config\\graph.json'

    data_keys_map = DataMap(map_path, data_keys_path)
    graph_tests = GraphTest.read_tests_info_file(graph_path)

    test_configs = []
    for data_info in data_keys_map.data_infos:
        data_tests = GraphTest.get_tests_for_file_key(data_info.data_key, graph_tests)
        tests_configs = [d.get_test_config(data_info, data_keys_map.data_infos) for d in data_tests]

        if not data_keys_map.settings.show_tests_not_ready_for_launch:
            tests_configs = [t_c for t_c in tests_configs if t_c['ready_for_launch']]

        test_configs.append({
            "file_key": data_info.data_key,
            "tests": tests_configs
        })

    with open('data/result.json', 'w+', encoding='utf-8') as file:
        json.dump(test_configs, file, ensure_ascii=False)
