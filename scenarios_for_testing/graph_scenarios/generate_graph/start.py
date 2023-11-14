from os import listdir
from os.path import join

from qa_qc_lib.graph.tools.generate_test_map_from_csv import generate_test_map_from_csv
from scenarios_for_testing.graph_scenarios import project_path

if __name__ == '__main__':
    # Список кодовых имён файлов
    data_keys_path = f'{project_path}qa_qc_lib\\graph\\config\\data_keys.txt'

    # Директория с .csv файлами
    csv_dir_path = 'graph_csvs'

    result_path = f'{project_path}qa_qc_lib\\graph\\config\\graph.json'

    with open(data_keys_path, encoding='utf-8') as file:
        good_keys = file.read().splitlines()

    csv_paths = [join(csv_dir_path, f) for f in listdir('graph_csvs')]

    generate_test_map_from_csv(csv_paths, result_path, good_keys)
