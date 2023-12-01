import dataclasses
import json

from qa_qc_lib.graph.data_map.DataMap import DataMap
from qa_qc_lib.graph.graph import Graph
from qa_qc_lib.graph.test_config.MainTestConfig import MainTestConfig
from scenarios_for_testing.graph_scenarios import project_path

if __name__ == '__main__':
    map_path = 'data/map.json'
    graph_path = f'{project_path}qa_qc_lib\\graph\\config\\graph.json'

    data_map = DataMap.read_map(map_path)
    graph = Graph(graph_path)

    launch_config = MainTestConfig.create_main_test_config(data_map, graph)

    with open("data/result.json", 'w+', encoding='utf-8') as file:
        json.dump(dataclasses.asdict(launch_config), file, ensure_ascii=False)

    exit(0)
