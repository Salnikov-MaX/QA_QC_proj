import dataclasses
import json

from dacite import from_dict

from qa_qc_lib.graph.test_config import generate_launch_config
from qa_qc_lib.graph.tools.data_map import read_map
from qa_qc_lib.graph.tools.graph import Graph
from scenarios_for_testing.graph_scenarios import project_path

if __name__ == '__main__':
    map_path = 'data/map.json'
    graph_path = f'{project_path}qa_qc_lib\\graph\\config\\graph.json'

    data_map = read_map(map_path)
    graph = Graph(graph_path)

    launch_config = generate_launch_config(data_map, graph)

    with open("data/result.json", 'w+', encoding='utf-8') as file:
        json.dump(dataclasses.asdict(launch_config), file, ensure_ascii=False)

    exit(0)
