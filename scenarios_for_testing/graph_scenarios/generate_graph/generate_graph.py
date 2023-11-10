from qa_qc_lib.graph.tools.generate_test_map_from_csv import generate_test_map_from_csv

from os import listdir
from os.path import isfile, join

if __name__ == '__main__':
    project_path = '/'
    data_keys_path = f'{project_path}\\qa_qc_lib\\graph\\config\\data_keys.txt'
    result_path = f'{project_path}\\qa_qc_lib\\graph\\config\\kern_graph.json'

    with open(data_keys_path, encoding='utf-8') as file:
        good_keys = file.read().splitlines()

    csv_paths = [f for f in listdir('graph_csvs') if isfile(join('graph_csvs', f))]

    generate_test_map_from_csv(csv_paths, result_path, good_keys)
