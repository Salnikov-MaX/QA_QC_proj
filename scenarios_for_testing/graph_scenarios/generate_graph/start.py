from os import listdir
from os.path import join

from qa_qc_lib.graph.tools.graph import Graph
from scenarios_for_testing.graph_scenarios import project_path

if __name__ == '__main__':
    # Список кодовых имён файлов
    data_keys_path = f'{project_path}qa_qc_lib\\graph\\config\\data_keys.txt'

    with open(data_keys_path, encoding='utf-8') as file:
        good_keys = file.read().splitlines()

    # Директория с .csv файлами
    csv_dir_path = 'graph_csvs'
    csv_paths = [join(csv_dir_path, f) for f in listdir(csv_dir_path)]

    result_path = f'{project_path}qa_qc_lib\\graph\\config\\graph.json'
    Graph.convert_graph_from_csv_to_json(csv_paths, result_path, good_keys)
